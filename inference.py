from __future__ import annotations
import threading
import time
from dataclasses import dataclass, replace
from typing import List, Optional
import mlx.core as mx
import configs
from apex_nadir_convolution import ApexNadirConvolution
from central import CentralModel
from diagnostics import Diagnostics
from experts import ExpertPool, ExpertOutput
from gating import GateModel, GateOutput, TripleKSelector, MaskingSchedule, SelectedExpert
from memory import RoutingMemory, SessionTracker
from splitter import (
    compute_xy,
    build_geography_batches,
    compute_x_expert_splits,
    compute_overlap_padding,
    get_available_ram_mb,
    prefetch_next_batch,
    ExpertFragment,
)
from tools import route_and_execute_tool
from bios import BIOS, route_bios_intent
from conversational_memory import ConversationalMemory
from code_executor import CodeExecutor
from self_correction import SelfCorrection
from k_velocity import KVelocity
from emotional import EmotionalCalibration
from predictive import PredictiveAssistant
from world_model import WorldModel
from vision_module import VisionSystem


@dataclass
class InferenceResult:
    output_text: str
    k_used: int
    experts_activated: List[int]
    timeline: str
    send_to_user: bool
    domain: str
    token_count: int
    reconstruction_entropy: float
    confidence: float
    mean_r_i: float
    x_next: int
    thermal_state: float
    ram_headroom_mb: float
    ssd_read_rate_mb: float
    emotional_context: str = ""
    memory_context: str = ""
    was_corrected: bool = False
class InferenceEngine:
    def __init__(
        self,
        gate: GateModel,
        expert_pool: ExpertPool,
        central: CentralModel,
        convolution: ApexNadirConvolution,
        routing_memory: RoutingMemory,
        session_tracker: SessionTracker,
        triple_k: TripleKSelector,
        masking_schedule: MaskingSchedule,
        bios: Optional[BIOS] = None,
        conv_memory: Optional[ConversationalMemory] = None,
        code_executor: Optional[CodeExecutor] = None,
        self_correction: Optional[SelfCorrection] = None,
        k_velocity: Optional[KVelocity] = None,
        emotional: Optional[EmotionalCalibration] = None,
        predictive: Optional[PredictiveAssistant] = None,
        world_model: Optional[WorldModel] = None,
        vision: Optional[VisionSystem] = None,
    ):
        self.gate = gate
        self.expert_pool = expert_pool
        self.central = central
        self.convolution = convolution
        self.routing_memory = routing_memory
        self.session_tracker = session_tracker
        self.triple_k = triple_k
        self.masking = masking_schedule
        self._batch_counter = 0
        self.diagnostics = Diagnostics()
        self._current_x = configs.X_MAX
        self._tokens_processed = 0

        # BRIAN capability stack
        self.bios = bios or BIOS()
        self.conv_memory = conv_memory
        self.code_executor = code_executor
        self.self_correction = self_correction
        self.k_velocity = k_velocity
        self.emotional = emotional
        self.predictive = predictive
        self.world_model = world_model
        self.vision = vision

    def _augment_input(self, input_text: str, min_experts: int) -> tuple:
        """Inject all BRIAN context (memory, emotional, world-model, BIOS, vision,
        code, tools, predictive) into the prompt and apply the K-velocity routing
        boost. Shared by run() and stream_reply() so both get identical context."""
        emotional_context = ""
        memory_context = ""

        if self.emotional and configs.EMOTIONAL_ENABLED:
            self.emotional.update_from_text(input_text)
            emotional_context = self.emotional.get_style_instruction()

        if self.conv_memory:
            memory_context = self.conv_memory.recall_context(input_text)

        if self.world_model and configs.WORLD_MODEL_ENABLED:
            self.world_model.observe(input_text)

        if self.bios and configs.BIOS_ENABLED:
            try:
                bios_result = route_bios_intent(input_text, self.bios)
                if bios_result:
                    input_text = f"[System Information]\n{bios_result}\n\nUser Question: {input_text}"
            except Exception:
                pass

        if self.vision and configs.VISION_ENABLED:
            try:
                vision_result = self._check_vision_intent(input_text)
                if vision_result:
                    input_text = f"[Vision]\n{vision_result}\n\nUser Question: {input_text}"
            except Exception:
                pass

        if self.code_executor:
            try:
                code_context = self._check_code_intent(input_text)
                if code_context:
                    input_text = f"[Code Execution Result]\n{code_context}\n\nUser Question: {input_text}"
            except Exception:
                pass

        try:
            tool_context = route_and_execute_tool(input_text)
            if tool_context:
                input_text = f"[Real-Time Information Context]\n{tool_context}\n\nUser Question: {input_text}"
        except Exception:
            pass

        context_parts = []
        if memory_context:
            context_parts.append(memory_context)
        if emotional_context:
            context_parts.append(emotional_context)
        if self.world_model and configs.WORLD_MODEL_ENABLED:
            wm_context = self.world_model.get_context_for_query(input_text)
            if wm_context:
                context_parts.append(wm_context)
        if self.predictive and configs.PREDICTIVE_ENABLED:
            pred_context = self.predictive.get_prediction_context()
            if pred_context:
                context_parts.append(pred_context)
        if context_parts:
            input_text = "\n".join(context_parts) + "\n\n" + input_text

        if self.k_velocity:
            suggested_k = self.k_velocity.suggest_k(self._domain_from_text_hint(input_text))
            if suggested_k > min_experts:
                min_experts = suggested_k

        return input_text, min_experts

    def stream_reply(self, input_text: str, on_sentence, conversation_id: str = "",
                     should_stop=None) -> str:
        """Low-latency voice path: same context as run(), central-only, streamed
        sentence-by-sentence to `on_sentence` so speech starts ~1s in. Returns the
        full reply. `should_stop()` (optional) lets the caller abort mid-stream
        (barge-in). Records to memory/k-velocity/predictive afterward."""
        augmented, _ = self._augment_input(input_text, 0)
        self.central.load()
        buffer = ""
        full = ""
        boundary = (".", "!", "?", "।", "\n")
        for delta in self.central.generate_stream(augmented):
            if should_stop is not None and should_stop():
                break
            buffer += delta
            full += delta
            # Flush complete sentences as they form.
            while any(b in buffer for b in boundary):
                idx = min(buffer.find(b) for b in boundary if b in buffer)
                sentence = buffer[: idx + 1].strip()
                buffer = buffer[idx + 1:]
                if sentence:
                    on_sentence(sentence)
                    if should_stop is not None and should_stop():
                        break
        if buffer.strip() and not (should_stop is not None and should_stop()):
            on_sentence(buffer.strip())

        # Post-inference recording (lightweight result; timeline A, no experts).
        result = InferenceResult(
            output_text=full, k_used=0, experts_activated=[], timeline="A",
            send_to_user=True, domain=self._domain_from_text_hint(input_text),
            token_count=len(full.split()), reconstruction_entropy=0.0,
            confidence=1.0, mean_r_i=0.0, x_next=self._current_x,
            thermal_state=0.0, ram_headroom_mb=0.0, ssd_read_rate_mb=0.0,
        )
        try:
            self._post_inference(result, input_text, conversation_id, allow_correction=False)
        except Exception:
            pass
        return full

    def run(
        self,
        input_text: str,
        send_to_user: bool = True,
        force_timeline_b: bool = False,
        force_timeline_a: bool = False,
        min_experts: int = 0,
        conversation_id: str = "",
    ) -> InferenceResult:
        if configs.DEPLOYMENT and not force_timeline_b:
            force_timeline_a = True

        if send_to_user:
            input_text, min_experts = self._augment_input(input_text, min_experts)

        self.gate.load()
        tokenizer = self.gate.tokenizer
        token_ids = tokenizer.encode(input_text)
        tokens = mx.array(token_ids)
        # Fused: one gate backbone pass yields both the routing decision and the
        # domain topography. Previously Timeline B ran the gate backbone twice
        # (forward here + look_ahead inside _timeline_b).
        gate_out, topo = self.gate.forward_with_topography(tokens)
        cluster_hit = self.routing_memory.lookup(gate_out.hidden_states)
        domain = self._domain_from_gate_output(gate_out)
        if force_timeline_a:
            k_floor = 0
        elif force_timeline_b:
            k_floor = max(1, min_experts)
        else:
            k_floor = min_experts
        selected_experts = self._select_experts_for_request(gate_out, cluster_hit, k_floor=k_floor)
        if force_timeline_a:
            return self._timeline_a(input_text, send_to_user, domain, len(token_ids), gate_out.confidence, conversation_id)
        if not force_timeline_b and self._is_timeline_a(gate_out):
            return self._timeline_a(input_text, send_to_user, domain, len(token_ids), gate_out.confidence, conversation_id)
        return self._timeline_b(
            input_text,
            tokens,
            gate_out,
            cluster_hit,
            send_to_user,
            selected_experts=selected_experts,
            default_domain=domain,
            min_experts=k_floor,
            conversation_id=conversation_id,
            topo=topo,
        )
    def _domain_from_gate_output(self, gate_out: GateOutput) -> str:
        domains = ["code", "reasoning", "knowledge", "general"]
        logits = gate_out.domain_logits
        if logits is None or logits.shape[0] == 0:
            return "general"
        vals = logits.tolist()
        if len(vals) < len(domains):
            return "general"
        return domains[int(mx.argmax(logits[:len(domains)]).item())]
    def _select_experts_for_request(self, gate_out: GateOutput, cluster_hit, k_floor: int = 0) -> List[SelectedExpert]:
        k_floor = max(0, min(configs.K_MAX, int(k_floor)))
        if cluster_hit is not None:
            cached_k = max(1, k_floor, int(cluster_hit.optimal_k))
            cached_k = min(configs.K_MAX, cached_k)
            selected = [
                SelectedExpert(expert_id=eid, distance_to_peak=0.0, domain="cached", is_alpha=False)
                for eid in cluster_hit.top_experts[:cached_k]
            ]
            if len(selected) >= cached_k:
                return selected
            fallback_gate = replace(gate_out, k_per_token=cached_k)
            for candidate in self.triple_k.select_experts(
                fallback_gate,
                self.session_tracker,
                self.masking,
                self._batch_counter,
            ):
                if len(selected) >= cached_k:
                    break
                if any(existing.expert_id == candidate.expert_id for existing in selected):
                    continue
                selected.append(candidate)
            return selected
        requested_k = min(configs.K_MAX, max(k_floor, int(gate_out.k_per_token)))
        if requested_k != gate_out.k_per_token:
            gate_out = replace(gate_out, k_per_token=requested_k)
        return self.triple_k.select_experts(gate_out, self.session_tracker, self.masking, self._batch_counter)
    def _is_timeline_a(self, gate_out: GateOutput) -> bool:
        return gate_out.confidence > configs.FAST_PATH_THRESHOLD
    def _all_candidate_fragments_below_nadir(self, tokens: mx.array, selected_experts: List[SelectedExpert]) -> bool:
        if not selected_experts:
            return False
        total_tokens = max(1, int(tokens.shape[0]))
        r_out_values = [
            max(float(configs.FRAGMENT_MIN), self.convolution.compute_r_out(sel.expert_id))
            for sel in selected_experts
        ]
        r_out_sum = sum(r_out_values) or float(len(r_out_values))
        for sel, r_out_i in zip(selected_experts, r_out_values):
            estimated_len = max(1, int(round(total_tokens * (r_out_i / r_out_sum))))
            if not self.convolution.check_nadir_floor(sel.expert_id, estimated_len):
                return False
        return True
    def _latest_diagnostics(self, ram_fallback: float = 0.0):
        if self.diagnostics.history:
            latest = self.diagnostics.history[-1]
            return self._current_x, latest.thermal_state, latest.ram_headroom_mb, latest.ssd_read_rate_mb
        return self._current_x, 0.0, ram_fallback, 0.0
    def _timeline_a(
        self,
        input_text: str,
        send_to_user: bool,
        domain: str,
        token_count: int,
        confidence: float,
        conversation_id: str = "",
    ) -> InferenceResult:
        self.central.load()
        output_text = self.central.generate(input_text)
        self.session_tracker.record_timeline_a(token_count)
        x_next, thermal, ram, ssd = self._latest_diagnostics()
        result = InferenceResult(
            output_text=output_text if send_to_user else "",
            k_used=0,
            experts_activated=[],
            timeline="A",
            send_to_user=send_to_user,
            domain=domain,
            token_count=token_count,
            reconstruction_entropy=0.0,
            confidence=confidence,
            mean_r_i=0.0,
            x_next=x_next,
            thermal_state=thermal,
            ram_headroom_mb=ram,
            ssd_read_rate_mb=ssd,
        )
        # Timeline A still records to memory/k-velocity/predictive (no self-correction —
        # the fast path has no experts to re-route).
        if send_to_user:
            result = self._post_inference(result, input_text, conversation_id, allow_correction=False)
        return result
    def _timeline_b(
        self,
        input_text: str,
        tokens: mx.array,
        gate_out: GateOutput,
        cluster_hit,
        send_to_user: bool,
        selected_experts: Optional[List[SelectedExpert]] = None,
        default_domain: str = "general",
        min_experts: int = 0,
        conversation_id: str = "",
        topo=None,
    ) -> InferenceResult:
        # topo is normally computed in the same backbone pass as gate_out
        # (forward_with_topography). Fall back to a dedicated pass only if a
        # caller didn't supply it.
        if topo is None:
            topo = self.gate.look_ahead(tokens)
        if selected_experts is None:
            selected_experts = self._select_experts_for_request(gate_out, cluster_hit)
        domain = max(topo.domain_proportions, key=topo.domain_proportions.get) if topo.domain_proportions else default_domain
        # If we have no experts but min_experts demands them, force-select
        # from the full pool instead of silently falling back to Central-only.
        if not selected_experts and min_experts > 0:
            forced_gate = replace(gate_out, k_per_token=min_experts)
            selected_experts = self.triple_k.select_experts(
                forced_gate, self.session_tracker, self.masking, self._batch_counter
            )
        if not selected_experts:
            self.central.load()
            output_text = self.central.generate(input_text)
            x_next, thermal, ram, ssd = self._latest_diagnostics()
            return InferenceResult(
                output_text=output_text if send_to_user else "",
                k_used=0,
                experts_activated=[],
                timeline="B",
                send_to_user=send_to_user,
                domain=domain,
                token_count=int(tokens.shape[0]),
                reconstruction_entropy=0.0,
                confidence=gate_out.confidence,
                mean_r_i=0.0,
                x_next=x_next,
                thermal_state=thermal,
                ram_headroom_mb=ram,
                ssd_read_rate_mb=ssd,
            )
        expert_ids = [se.expert_id for se in selected_experts]
        r_out_mean = self.convolution.compute_r_out_mean(expert_ids)
        available_ram = get_available_ram_mb()
        total_tokens = tokens.shape[0]
        if available_ram < configs.EXPERT_RAM_MB:
            self.central.load()
            output_text = self.central.generate(input_text)
            x_next, thermal, ram, ssd = self._latest_diagnostics(available_ram)
            return InferenceResult(
                output_text=output_text if send_to_user else "",
                k_used=0,
                experts_activated=[],
                timeline="B",
                send_to_user=send_to_user,
                domain=domain,
                token_count=int(total_tokens),
                reconstruction_entropy=0.0,
                confidence=gate_out.confidence,
                mean_r_i=0.0,
                x_next=x_next,
                thermal_state=thermal,
                ram_headroom_mb=ram,
                ssd_read_rate_mb=ssd,
            )
        geometry = compute_xy(
            max(1, total_tokens),
            max(1.0, r_out_mean),
            available_ram,
            x_override=self._current_x,
        )
        batches = build_geography_batches(tokens, topo.domain_map, geometry.Y)
        all_expert_outputs: List[ExpertOutput] = []
        previous_expert_ids: set = set()
        prefetch_event: Optional[threading.Event] = None
        prefetch_thread: Optional[threading.Thread] = None
        for i, batch in enumerate(batches):
            batch_start = time.time()
            if len(batch.token_indices) == 0:
                continue
            if prefetch_event is not None:
                prefetch_event.wait()
                if prefetch_thread is not None:
                    prefetch_thread.join(timeout=0)
                prefetch_event = None
                prefetch_thread = None
            x_used = self._current_x
            fragments = compute_x_expert_splits(batch, selected_experts, x_used, self.convolution)
            batch_expert_ids = [f.expert_id for f in fragments if not f.below_nadir]
            if not batch_expert_ids:
                for fragment in fragments:
                    if fragment.below_nadir:
                        self._shadow_audit(fragment, batch.tokens, fragment.expert_id)
                self._tokens_processed += len(batch.token_indices)
                self._current_x = self.diagnostics.update(
                    tokens_processed=self._tokens_processed,
                    time_in_bound=time.time() - batch_start,
                    x_used=x_used,
                    k_used=0,
                )
                continue
            ids_to_unload = list(previous_expert_ids - set(batch_expert_ids))
            self.expert_pool.unload_experts(ids_to_unload, keep_buffer=set(batch_expert_ids) & previous_expert_ids)
            loaded_ids = set(self.expert_pool.loaded_experts)
            missing_expert_ids = [eid for eid in batch_expert_ids if eid not in loaded_ids]
            if missing_expert_ids:
                self.expert_pool.load_experts(missing_expert_ids)
            loaded_ids = set(self.expert_pool.loaded_experts)
            current_selected_experts = [
                se for se in selected_experts if se.expert_id in batch_expert_ids and se.expert_id in loaded_ids
            ]
            if not current_selected_experts:
                self._tokens_processed += len(batch.token_indices)
                self._current_x = self.diagnostics.update(
                    tokens_processed=self._tokens_processed,
                    time_in_bound=time.time() - batch_start,
                    x_used=x_used,
                    k_used=0,
                )
                continue
            batch_expert_ids = [se.expert_id for se in current_selected_experts]
            fragments = compute_x_expert_splits(batch, current_selected_experts, x_used, self.convolution)
            if i + 1 < len(batches):
                next_batch = batches[i + 1]
                next_fragments = compute_x_expert_splits(
                    next_batch,
                    selected_experts,
                    self._current_x,
                    self.convolution,
                )
                next_expert_ids = [f.expert_id for f in next_fragments if not f.below_nadir]
                if next_expert_ids:
                    prefetch_event = threading.Event()
                    prefetch_thread = threading.Thread(
                        target=prefetch_next_batch,
                        args=(self.expert_pool, next_expert_ids, prefetch_event),
                        daemon=True,
                    )
                    prefetch_thread.start()
            for fragment in fragments:
                if fragment.below_nadir:
                    self._shadow_audit(fragment, batch.tokens, fragment.expert_id)
                    continue
                if fragment.expert_id not in loaded_ids:
                    continue
                eo = self.expert_pool.expert_forward(fragment.expert_id, fragment.tokens)
                all_expert_outputs.append(eo)
            self._tokens_processed += len(batch.token_indices)
            x_next = self.diagnostics.update(
                tokens_processed=self._tokens_processed,
                time_in_bound=time.time() - batch_start,
                x_used=x_used,
                k_used=len(batch_expert_ids),
            )
            self._current_x = x_next
            previous_expert_ids = set(batch_expert_ids)
        if prefetch_event is not None:
            prefetch_event.wait()
            if prefetch_thread is not None:
                prefetch_thread.join(timeout=0)
        if previous_expert_ids:
            self.expert_pool.unload_experts(list(previous_expert_ids))
        self.central.load()
        expert_data = [
            {"expert_id": eo.expert_id, "output_text": eo.output_text, "hidden_states": eo.hidden_states, "wall_time": eo.wall_time}
            for eo in all_expert_outputs
        ]
        central_out = self.central.forward(input_text, expert_data, send_to_user=send_to_user)
        r_i_scores: List[float] = []
        for eo in all_expert_outputs:
            r_i = self.central.compute_r_i(eo.hidden_states, central_out.contribution_hidden, eo.wall_time)
            r_out = self.convolution.compute_r_out(eo.expert_id)
            anchor = self.expert_pool.get_historical_anchor(eo.expert_id)
            tkl = self.central.compute_tkl(r_i, r_out, anchor, eo.wall_time)
            self.session_tracker.record_activation(eo.expert_id, eo.token_count, r_i, eo.wall_time, tkl, domain)
            self.expert_pool.update_domain_score(eo.expert_id, domain, r_i)
            self.central.update_r_t(eo.expert_id, eo.token_count, eo.wall_time, self.convolution)
            r_i_scores.append(r_i)
        activated = [eo.expert_id for eo in all_expert_outputs]
        mean_r_i = sum(r_i_scores) / len(r_i_scores) if r_i_scores else 0.0
        x_next, thermal, ram, ssd = self._latest_diagnostics(available_ram)
        self._batch_counter += 1
        if send_to_user:
            output_text = central_out.synthesis_text
            if not output_text.strip():
                output_text = self.central.generate(input_text)
        else:
            output_text = ""
        result = InferenceResult(
            output_text=output_text,
            k_used=len(set(activated)),
            experts_activated=activated,
            timeline="B",
            send_to_user=send_to_user,
            domain=domain,
            token_count=int(total_tokens),
            reconstruction_entropy=central_out.reconstruction_entropy,
            confidence=gate_out.confidence,
            mean_r_i=mean_r_i,
            x_next=x_next,
            thermal_state=thermal,
            ram_headroom_mb=ram,
            ssd_read_rate_mb=ssd,
        )
        # --- Post-inference BRIAN hooks ---
        if send_to_user:
            result = self._post_inference(result, input_text, conversation_id)
        return result

    def _post_inference(
        self,
        result: InferenceResult,
        input_text: str,
        conversation_id: str,
        allow_correction: bool = True,
    ) -> InferenceResult:
        # Self-correction — never in deployment (fast path), never during a
        # correction retry (the retry sets _in_correction to avoid recursion).
        if (
            allow_correction
            and self.self_correction
            and configs.SELF_CORRECTION_ENABLED
            and not configs.DEPLOYMENT
            and not getattr(self, "_in_correction", False)
        ):
            if self.self_correction.needs_correction(result.confidence, result.reconstruction_entropy, result.mean_r_i):
                self._in_correction = True
                try:
                    correction = self.self_correction.run_correction_loop(self, input_text, result)
                    if correction.corrected:
                        result = replace(result, output_text=correction.final_output, was_corrected=True)
                finally:
                    self._in_correction = False

        # Skip side-effect recording when this call is itself a correction retry
        # (the outer user turn records once; retries must not double-count).
        if getattr(self, "_in_correction", False):
            return result

        # K-Velocity recording
        if self.k_velocity:
            self.k_velocity.record_event(
                expert_ids=result.experts_activated,
                domain=result.domain,
                k_used=result.k_used,
                confidence=result.confidence,
                mean_r_i=result.mean_r_i,
                query_text=input_text[:200],
            )

        # Conversational memory storage
        if self.conv_memory and conversation_id:
            self.conv_memory.store_turn(conversation_id, "user", input_text, result.domain)
            if result.output_text:
                self.conv_memory.store_turn(conversation_id, "assistant", result.output_text, result.domain)

        # Predictive recording
        if self.predictive and configs.PREDICTIVE_ENABLED:
            quality = result.confidence * 0.5 + result.mean_r_i * 0.5
            self.predictive.record_interaction(result.domain, result.timeline, input_text[:100], quality)

        return result

    def _check_code_intent(self, text: str) -> Optional[str]:
        import re
        q = text.lower()
        if not any(kw in q for kw in ["run this", "execute", "run the code", "run code", "```"]):
            return None
        code_match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if not code_match:
            return None
        code = code_match.group(1).strip()
        lang = self.code_executor.detect_language(code)
        result = self.code_executor.execute(code, lang)
        if result.returncode == 0:
            return f"[{lang}] Exit 0\n{result.stdout[:1000]}"
        return f"[{lang}] Exit {result.returncode}\nstdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"

    def _check_vision_intent(self, text: str) -> Optional[str]:
        import re
        q = text.lower()
        screen_kw = ["my screen", "on screen", "what do you see", "look at my screen",
                     "what's on my screen", "whats on my screen", "see my screen", "read my screen"]
        if any(kw in q for kw in screen_kw):
            info = self.vision.see_screen()
            if "error" in info:
                return f"(Could not capture screen: {info['error']})"
            parts = [f"Visual content: {info.get('visual_content', '?')}"]
            text_seen = info.get("text_on_screen", "")
            if text_seen and text_seen != "(no text detected)":
                parts.append(f"Text on screen:\n{text_seen[:1200]}")
            if info.get("changes"):
                parts.append(info["changes"])
            return "\n".join(parts)
        # "look at <path>" / "read image <path>"
        img_match = re.search(r"(?:look at|read|describe|see)\s+(?:the\s+)?(?:image|picture|photo|file)?\s*([~/\w.\-]+\.(?:png|jpg|jpeg|heic|gif|bmp|tiff))", q)
        if img_match:
            info = self.vision.read_image(img_match.group(1))
            if "error" in info:
                return f"(Could not read image: {info['error']})"
            parts = [f"Visual content: {info.get('visual_content', '?')}"]
            if info.get("text_extracted") and info["text_extracted"] != "(no text)":
                parts.append(f"Text in image:\n{info['text_extracted'][:1200]}")
            return "\n".join(parts)
        return None

    def _domain_from_text_hint(self, text: str) -> str:
        q = text.lower()
        if any(kw in q for kw in ["code", "function", "python", "javascript", "program"]):
            return "code"
        if any(kw in q for kw in ["math", "equation", "calculate", "integral"]):
            return "reasoning"
        if any(kw in q for kw in ["who", "what", "when", "where", "history", "science"]):
            return "knowledge"
        return "general"

    def _shadow_audit(self, fragment: ExpertFragment, context: mx.array, expert_id: int):
        padded = compute_overlap_padding(fragment, context)
        if padded.grad_mask.shape[0] != padded.padded_tokens.shape[0]:
            raise AssertionError(
                f"Expert {expert_id}: shadow audit mask/token length mismatch "
                f"({padded.grad_mask.shape[0]} vs {padded.padded_tokens.shape[0]})."
            )
        return padded

if __name__ == "__main__":
    import argparse
    from main import boot_system, DeadTimeState
    
    parser = argparse.ArgumentParser(description="Run Sturnus inference test.")
    parser.add_argument("--deployment", action="store_true", help="Run in deployment mode.")
    args = parser.parse_args()
    
    if args.deployment:
        configs.DEPLOYMENT = True
        
    print("Booting Sturnus system...")
    components = boot_system()
    dead_state = DeadTimeState()
    
    test_queries = [
        "What is quantum entanglement and how does it relate to Bell's theorem?",
        "Write a Python function that implements merge sort with O(n log n) complexity",
        "Explain the causes and consequences of the French Revolution"
    ]
    
    print(f"\nRunning inference tests (deployment mode = {configs.DEPLOYMENT})...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: '{query}' ---")
        start_time = time.time()
        result = components.inference_engine.run(query)
        latency = (time.time() - start_time) * 1000.0
        print(f"Output: {result.output_text}")
        print(f"Timeline: {result.timeline}")
        print(f"Experts Activated (K): {result.k_used} {result.experts_activated}")
        print(f"Mean R_i: {result.mean_r_i:.4f}")
        print(f"Latency: {latency:.1f} ms")
