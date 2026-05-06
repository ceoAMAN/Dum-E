from __future__ import annotations
import threading
import time
from dataclasses import dataclass
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
    def run(
        self,
        input_text: str,
        send_to_user: bool = True,
        force_timeline_b: bool = False,
        force_timeline_a: bool = False,
    ) -> InferenceResult:
        self.gate.load()
        tokenizer = self.gate.tokenizer
        token_ids = tokenizer.encode(input_text)
        tokens = mx.array(token_ids)
        gate_out = self.gate.forward(tokens)
        cluster_hit = self.routing_memory.lookup(gate_out.hidden_states)
        domain = self._domain_from_gate_output(gate_out)
        selected_experts = self._select_experts_for_request(gate_out, cluster_hit)
        if force_timeline_a:
            return self._timeline_a(input_text, send_to_user, domain, len(token_ids), gate_out.confidence)
        if not force_timeline_b and self._is_timeline_a(gate_out):
            return self._timeline_a(input_text, send_to_user, domain, len(token_ids), gate_out.confidence)
        return self._timeline_b(
            input_text,
            tokens,
            gate_out,
            cluster_hit,
            send_to_user,
            selected_experts=selected_experts,
            default_domain=domain,
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
    def _select_experts_for_request(self, gate_out: GateOutput, cluster_hit) -> List[SelectedExpert]:
        if cluster_hit is not None:
            cached_k = max(1, int(cluster_hit.optimal_k))
            return [
                SelectedExpert(expert_id=eid, distance_to_peak=0.0, domain="cached", is_alpha=False)
                for eid in cluster_hit.top_experts[:cached_k]
            ]
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
    ) -> InferenceResult:
        self.central.load()
        output_text = self.central.generate(input_text)
        self.session_tracker.record_timeline_a(token_count)
        x_next, thermal, ram, ssd = self._latest_diagnostics()
        return InferenceResult(
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
    def _timeline_b(
        self,
        input_text: str,
        tokens: mx.array,
        gate_out: GateOutput,
        cluster_hit,
        send_to_user: bool,
        selected_experts: Optional[List[SelectedExpert]] = None,
        default_domain: str = "general",
    ) -> InferenceResult:
        topo = self.gate.look_ahead(tokens)
        if selected_experts is None:
            selected_experts = self._select_experts_for_request(gate_out, cluster_hit)
        domain = max(topo.domain_proportions, key=topo.domain_proportions.get) if topo.domain_proportions else default_domain
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
        return InferenceResult(
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
    def _shadow_audit(self, fragment: ExpertFragment, context: mx.array, expert_id: int):
        padded = compute_overlap_padding(fragment, context)
        if padded.grad_mask.shape[0] != padded.padded_tokens.shape[0]:
            raise AssertionError(
                f"Expert {expert_id}: shadow audit mask/token length mismatch "
                f"({padded.grad_mask.shape[0]} vs {padded.padded_tokens.shape[0]})."
            )
        return padded
