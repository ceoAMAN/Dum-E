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

    def stream_reply(self, input_text: str, on_sentence, conversation_id: str = "",
                     should_stop=None) -> str:
        """Low-latency streaming path: central-only, streamed sentence-by-sentence
        to `on_sentence` so output starts ~1s in. Returns the full reply.
        `should_stop()` (optional) lets the caller abort mid-stream."""
        augmented = input_text
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
                # Experts always hand Central both channels — hidden states + a
                # REAL generated analysis (no echo). Generation length is governed
                # by apex-nadir (R_out when calibrated, else the safety valve),
                # separate from the input fragment size.
                eo = self.expert_pool.expert_forward(
                    fragment.expert_id, fragment.tokens,
                    generate_text=True,
                    max_tokens=self.convolution.generation_length(fragment.expert_id),
                )
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
        # forward() is the self-supervision measurement pass (contribution_hidden,
        # r_i, reconstruction entropy). It is never the user reply now — the reply
        # is generated below with expert text injected — so skip its lm_head token.
        central_out = self.central.forward(input_text, expert_data, send_to_user=False)
        r_i_scores: List[float] = []
        for eo in all_expert_outputs:
            r_i = self.central.compute_r_i(eo.hidden_states, central_out.contribution_hidden, eo.wall_time, synthesis_hidden=central_out.synthesis_hidden)
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
            # Deployed reply: condition Central's generation on the experts' actual
            # generated analyses so the MoE pipeline reaches the user-facing answer
            # (audit A.2.4). No expert text → plain Central-only generation.
            expert_texts = [eo.output_text for eo in all_expert_outputs if eo.output_text.strip()]
            output_text = self.central.generate(input_text, expert_context=expert_texts or None)
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
        return result

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
