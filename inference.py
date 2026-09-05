from __future__ import annotations
import math
import threading
import time
from dataclasses import dataclass, replace
from typing import List, Optional
import mlx.core as mx
import configs
from apex_nadir_convolution import ApexNadirConvolution, probe_sizes
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
    schedule_by_expert,
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
        maml=None,
    ):
        self.gate = gate
        self.expert_pool = expert_pool
        self.central = central
        self.convolution = convolution
        self.routing_memory = routing_memory
        self.session_tracker = session_tracker
        self.triple_k = triple_k
        self.masking = masking_schedule
        # MAML adapts the L_gate term weights and PERSISTS them (main.session_reset
        # -> maml.save()), but nothing read them back: every consumer hardcoded
        # configs.LAMBDA_INIT, so the adapted values were computed, written to
        # disk and ignored. An adaptive quantity with no reader is this system's
        # signature bug, and it was reintroduced by the gate wiring itself.
        self.maml = maml
        self._batch_counter = 0
        self.diagnostics = Diagnostics()
        # Start from what the MACHINE can hold, not the ceiling: X_MAX is a soft
        # cap, and opening at 6 resident experts is the exact pattern that
        # aborted Metal on 16GB before any memory had been measured.
        from splitter import experts_per_batch
        self._current_x = max(configs.X_MIN, min(configs.X_MAX, experts_per_batch()))
        self._tokens_processed = 0
        # Central's capacity cap is re-probed every E inputs (E = pool size, read
        # live). BOTH timelines count toward the counter: Central sees the
        # original input on A and original+expert on B, so either is a valid
        # capacity sample. The cap has no effect on a Timeline B request — B is
        # already running experts — but measuring on B too keeps it current
        # instead of only tracking the small-input traffic.
        self._inputs_seen = 0
        self._capacity_samples: List[tuple] = []

    def run(
        self,
        input_text: str,
        send_to_user: bool = True,
        force_timeline_b: bool = False,
        force_timeline_a: bool = False,
        min_experts: int = 0,
        target_ids: Optional[List[int]] = None,
    ) -> InferenceResult:
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
        # Counts on BOTH timelines — see note_input. target_ids arrive only from a
        # training caller, so at inference this advances the counter and nothing more.
        self.note_input(input_text, target_ids=target_ids)
        if force_timeline_a:
            return self._timeline_a(input_text, send_to_user, domain, len(token_ids), gate_out.confidence)
        if not force_timeline_b and self._is_timeline_a(gate_out, len(token_ids), cluster_hit):
            return self._timeline_a(input_text, send_to_user, domain, len(token_ids), gate_out.confidence)
        return self._timeline_b(
            input_text,
            tokens,
            gate_out,
            cluster_hit,
            send_to_user,
            selected_experts=selected_experts,
            default_domain=domain,
            min_experts=k_floor,
            topo=topo,
            target_ids=target_ids,
        )
    def note_input(self, input_text: str, target_ids: Optional[List[int]] = None) -> Optional[float]:
        """Count one input toward the every-E capacity re-probe, and fire it when
        the counter comes round. Returns the new cap when it re-probed, else None.

        Called on EVERY input regardless of timeline. `target_ids` — the real
        continuation — is what makes an input usable as a capacity sample, since
        the probe measures cross-entropy against true text. Without targets the
        input still counts toward the cadence but contributes no sample, so at
        pure inference the probe simply never fires: there is no known-correct
        continuation for a live question to measure against.

        The buffer holds at most E samples, matching the cadence, so each probe
        reads the most recent window rather than the whole history."""
        from training import probe_central_capacity
        E = max(1, int(configs.EXPERT_POOL_SIZE))
        self._inputs_seen += 1
        if target_ids:
            self._capacity_samples.append((input_text, list(target_ids)))
            if len(self._capacity_samples) > E:
                self._capacity_samples = self._capacity_samples[-E:]
        if self._inputs_seen % E != 0 or len(self._capacity_samples) < 2:
            return None
        try:
            result = probe_central_capacity(self.central, self._capacity_samples)
        except Exception as e:
            print(f"[warn] central capacity probe failed: {e}")
            return None
        cap = float(result.get("min_cap", 0.0)) if result else 0.0
        if cap > 0.0:
            prev = self.convolution.central_min_cap
            self.convolution.central_min_cap = cap
            print(f"[capacity] input {self._inputs_seen}: central_min_cap {prev:.1f} -> {cap:.1f} "
                  f"(apex {result.get('apex', 0):.0f}, {int(result.get('probes', 0))} probes)")
            return cap
        return None

    def _domain_from_gate_output(self, gate_out: GateOutput) -> str:
        # Third copy of the domain list lived here, so adding a domain in
        # gating.DOMAINS silently left this one stale. Single source of truth.
        from gating import DOMAINS as domains
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
    def _is_timeline_a(self, gate_out: GateOutput, token_count: int, cluster_hit=None) -> bool:
        """Timeline A = Central alone. Two rules, in order.

        1. UNKNOWN input -> always Timeline B. A routing-memory miss means
           nothing like this has been seen, so the experts run: that is the case
           they exist for, regardless of size.
        2. Otherwise A iff `token_count < central_min_cap` — Central's OWN
           measured apex-nadir floor (training.probe_central_capacity). Below it
           an input is too small to spread across experts meaningfully, so
           Central handles it solo; above it, Timeline B.

        This replaces two bad rules. `DEPLOYMENT -> force Timeline A` sent every
        production query down the Central-only path unconditionally, which is why
        the MoE never reached a user. And `confidence > FAST_PATH_THRESHOLD` keyed
        the decision on entropy over a z-scored slice of hidden state — a number
        that tracks nothing about whether an input actually needs experts.
        Confidence survives only as the fallback until the cap has been probed.

        Timeline A still queues a shadow Timeline B (main.dead_time_orchestrator),
        so Central keeps its synthesiser skill and the experts do not decay from
        disuse."""
        if cluster_hit is None:
            return False
        cap = float(getattr(self.convolution, "central_min_cap", 0.0) or 0.0)
        if cap > 0.0:
            return int(token_count) < cap
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
        topo=None,
        target_ids: Optional[List[int]] = None,
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
        fragment_by_expert: dict = {}
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
            # OOM GUARD. can_fit_expert() reads the LAST batch's observed peak
            # against the usable ceiling; it had no caller, so the governor's
            # only actual safety output was never consulted. It returns True
            # while unmeasured, so this cannot bite before there is evidence.
            if not self.diagnostics.can_fit_expert():
                print(f"[mem] last peak at {self.diagnostics.peak_util():.0%} of usable "
                      f"(fallback at {configs.MEM_FALLBACK_FRAC:.0%}) — stopping at batch {i} "
                      f"of {len(batches)}; synthesising on the {len(all_expert_outputs)} "
                      f"expert output(s) already collected")
                break
            # Tombstone BEFORE the risky work. A Metal OOM is an uncatchable
            # abort, so nothing downstream of the crash can record it; the marker
            # has to already be on disk when the process dies. Cleared once the
            # cycle survives, and claimed at the next boot if it isn't.
            self.diagnostics.arm_oom_watch(x_used)
            fragments = compute_x_expert_splits(batch, selected_experts, x_used, self.convolution,
                                                tokenizer=self.gate.tokenizer)
            # The nadir is an ALLOCATION signal, not a gate: scarce tokens mean
            # FEWER experts (k = T/ALLOC(T)), never zero. This used to filter on
            # `not f.below_nadir`, which dropped every fragment shorter than
            # FRAGMENT_MIN=32 — and since R_out was pinned at 32, a question
            # needed >= 32*k tokens for a single expert to survive. A 16-token
            # question ran ZERO experts and Timeline B silently became Timeline A.
            batch_expert_ids = [f.expert_id for f in fragments]
            if not batch_expert_ids:
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
            # ROLLING RESIDENCY. Do NOT bulk-load the batch and then discard
            # whatever did not fit. load_experts() stops at _max_loaded (6), so
            # with k > 6 every expert past the sixth was silently dropped right
            # here — k was capped by how many models fit AT ONCE, and the batch
            # quietly ran fewer experts than were selected. Experts are now loaded
            # one at a time in the fragment loop below and the pool's LRU evicts a
            # FINISHED one to make room for the next, so k is bounded by RAM over
            # TIME instead. Selection is no longer filtered by residency.
            current_selected_experts = [
                se for se in selected_experts if se.expert_id in batch_expert_ids
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
            fragments = compute_x_expert_splits(batch, current_selected_experts, x_used, self.convolution,
                                                tokenizer=self.gate.tokenizer)
            if i + 1 < len(batches):
                next_batch = batches[i + 1]
                next_fragments = compute_x_expert_splits(
                    next_batch,
                    selected_experts,
                    self._current_x,
                    self.convolution,
                    tokenizer=self.gate.tokenizer,
                )
                next_expert_ids = [f.expert_id for f in next_fragments]
                if next_expert_ids:
                    prefetch_event = threading.Event()
                    prefetch_thread = threading.Thread(
                        target=prefetch_next_batch,
                        args=(self.expert_pool, next_expert_ids, prefetch_event),
                        daemon=True,
                    )
                    prefetch_thread.start()
            # Order the work so each expert's fragments run back-to-back: it
            # loads once, sees everything assigned to it, and only then makes way.
            # Token order would make it load, run, get evicted and load again.
            for fragment in schedule_by_expert(fragments, resident=loaded_ids):
                # Load on demand instead of skipping. schedule_by_expert already
                # yields resident experts first and, within that, the largest
                # token runs first, and it keeps each expert's fragments
                # contiguous — so an expert loads once, sees everything assigned
                # to it, and only then makes way. Loading here rather than
                # skipping is what turns that ordering into an actual rotation.
                if fragment.expert_id not in self.expert_pool.loaded_experts:
                    self.expert_pool.load_experts([fragment.expert_id])
                    if fragment.expert_id not in self.expert_pool.loaded_experts:
                        continue   # load genuinely failed — skip it, do not stall
                # No below_nadir skip: a short fragment is a legitimate (if weak)
                # contribution, and dropping it is what made the expert pool
                # unreachable for normal-length questions.
                # Experts always hand Central both channels — hidden states + a
                # REAL generated analysis (no echo). Generation length is governed
                # by apex-nadir (R_out when calibrated, else the safety valve),
                # separate from the input fragment size.
                eo = self.expert_pool.expert_forward(
                    fragment.expert_id, fragment.tokens,
                    generate_text=True,
                    max_tokens=self.convolution.generation_length(fragment.expert_id,
                                                                  total_tokens=int(total_tokens)),
                    allocated_tokens=fragment.allocated_len,
                    question=input_text,
                    domain=fragment.domain_label,
                )
                all_expert_outputs.append(eo)
                # Keep each expert's fragment: spiderweb needs a real forward pass
                # to attach its gradient to, and the fragment is what that expert
                # actually worked on this batch.
                fragment_by_expert[eo.expert_id] = fragment.tokens
            self._tokens_processed += len(batch.token_indices)
            # Decide the expert count from the STANDING scores — what the pool
            # was worth coming into this batch — not from scores this batch is
            # about to produce. Those need Central's forward, which needs every
            # expert's output, so they do not exist yet; and choosing x from
            # results that x itself produced would be circular. Prior evidence is
            # both the only thing available and the causally correct thing to use.
            # Keep the ids alongside the scores: the swap term needs to know WHICH
            # experts the top-x are, not just how many, or a fully-cold top-x
            # scores as free whenever the resident count happens to be larger.
            standing_pairs = sorted(
                ((e, v) for e, v in self.session_tracker.expert_tkl.items() if v > 0.0),
                key=lambda kv: kv[1], reverse=True,
            )
            standing = [v for _, v in standing_pairs]
            standing_ids = [e for e, _ in standing_pairs]
            x_next = self.diagnostics.update(
                tokens_processed=self._tokens_processed,
                time_in_bound=time.time() - batch_start,
                x_used=x_used,
                k_used=len(batch_expert_ids),
                ranked_tkl=standing,
                total_tokens=int(total_tokens),
                tokens_left=max(0, int(total_tokens) - self._tokens_processed),
                alloc_target=self.convolution.allocation(int(total_tokens)),
                resident=len(self.expert_pool.loaded_experts),
                ranked_ids=standing_ids,
                resident_ids=list(self.expert_pool.loaded_experts.keys()),
            )
            self._current_x = x_next
            previous_expert_ids = set(batch_expert_ids)
            self.diagnostics.disarm_oom_watch()      # cycle survived at this x
        if prefetch_event is not None:
            prefetch_event.wait()
            if prefetch_thread is not None:
                prefetch_thread.join(timeout=0)
        # DO NOT unload here. Everything downstream of this point — the r_i
        # measurement, composite TKL, the expert gradient step and the apex-nadir
        # refresh — needs these experts RESIDENT. Evicting first made all of them
        # no-ops: _apply_expert_learning filters `tkl_by_expert` against
        # loaded_experts and found nothing, so apply_expert_gradients was called
        # ZERO times on a batch where five experts ran and were scored; and
        # _refresh_apex_nadir reported "probing 0/N experts (rest not resident)"
        # on every batch, which is why ALLOC was never fitted and R_out sat at
        # FRAGMENT_MIN for the whole pool.
        #
        # The eviction was buying nothing anyway: experts_per_batch() already
        # subtracts Central and the gate from the RAM budget, so the concurrency
        # cap was computed on the assumption that they coexist. Central is also
        # normally resident by now and load() is idempotent. Unload moved to
        # after the learning pass.
        self.central.load()
        expert_data = [
            {"expert_id": eo.expert_id, "output_text": eo.output_text, "hidden_states": eo.hidden_states, "wall_time": eo.wall_time}
            for eo in all_expert_outputs
        ]
        # forward() is the self-supervision measurement pass (contribution_hidden,
        # r_i, reconstruction entropy). It is never the user reply now — the reply
        # is generated below with expert text injected — so skip its lm_head token.
        central_out = self.central.forward(input_text, expert_data, send_to_user=False)
        # Peer disagreement BEFORE the activations are recorded — it is an input to
        # composite TKL's no_halluc term, and it costs nothing extra: every expert
        # worked a fragment of the same input, so their hidden states are already
        # comparable and already computed.
        disagreement = self.expert_pool.peer_disagreement(all_expert_outputs)
        novel_input = cluster_hit is None      # routing-memory miss = never seen before
        # R_i post-synthesis, as a weighted sum over the whole batch: alignment +
        # non-hallucination + speed (+ grounded perfection when a real
        # continuation exists). Weights derived from each component's spread
        # across these experts, same rule as composite TKL.
        r_i_by_expert = self.central.compute_r_i_batch(
            [{"expert_id": eo.expert_id, "hidden_states": eo.hidden_states,
              "wall_time": eo.wall_time} for eo in all_expert_outputs],
            central_out.contribution_hidden, central_out.synthesis_hidden,
            disagreement=disagreement,
        )
        # GROUNDED OVERRIDE. compute_r_i_batch scores cosine against Central's
        # OWN hidden state, so an expert that confidently agrees with a
        # hallucinated synthesis scores well — the circular reward. When a real
        # continuation exists, replace those scores with the leave-one-out delta
        # measured against text neither model wrote. Sampled, not exhaustive:
        # leave-one-out over k experts costs k+1 Central forwards, and the signal
        # only has to be periodic to keep routing off the throughput heuristic.
        if target_ids:
            grounded = self._grounded_r_i_override(input_text, all_expert_outputs, target_ids)
            if grounded:
                r_i_by_expert.update(grounded)
        r_i_scores: List[float] = []
        tkl_by_expert: dict = {}
        for eo in all_expert_outputs:
            r_i = r_i_by_expert.get(eo.expert_id, 0.0)
            r_out = self.convolution.compute_r_out(eo.expert_id, total_tokens=int(total_tokens))
            anchor = self.expert_pool.get_historical_anchor(eo.expert_id)
            tkl = self.central.compute_tkl(r_i, r_out, anchor, eo.wall_time)
            tkl_by_expert[eo.expert_id] = tkl
            self.session_tracker.record_activation(
                eo.expert_id, eo.token_count, r_i, eo.wall_time, tkl, domain,
                disagreement=disagreement.get(eo.expert_id), novel=novel_input,
            )
            self.expert_pool.update_domain_score(eo.expert_id, domain, r_i)
            self.central.update_r_t(eo.expert_id, eo.token_count, eo.wall_time, self.convolution)
            # Probe record for the apex-nadir convolution: the expert was given
            # `token_count` tokens (the allocated share, never the word-boundary
            # padding) out of an input of `total_tokens`, and scored r_i.
            self.convolution.record_probe(eo.expert_id, int(total_tokens), int(eo.token_count), r_i,
                                          wall_time=eo.wall_time)
            r_i_scores.append(r_i)
        # AFTER the output: recompute composite TKL over everyone who ran and make
        # it THE ranking signal. compute_tkl's r_out*(r_i/c_e)*anchor multiplies
        # tokens^2 by a score by seconds — allocation size dominates it — so it is
        # kept only as the per-activation record that feeds domain means. Ranking
        # (spiderweb tiers, refresh selection) uses the composite: seven measured
        # components, missing ones dropped rather than zeroed, shrunk toward the
        # pool mean so two activations cannot outrank fifty.
        active_ids = [eo.expert_id for eo in all_expert_outputs]
        # Bind unconditionally. This was assigned only inside `if active_ids:`
        # while being READ further down at _apply_gate_learning, so a batch that
        # produced no expert output crashed the whole request on an unbound
        # local instead of degrading to Central-only.
        composite: dict = {}
        if active_ids:
            composite = self.session_tracker.composite_tkl_pool(active_ids, domain)
            for eid, c in composite.items():
                if c.get("n"):
                    tkl_by_expert[eid] = float(c["overall"])
                    self.session_tracker.expert_tkl[eid] = float(c["overall"])
        # Spiderweb runs on EVERY batch: the better half of the active set pulls
        # the worse half toward it. Dormant experts (no checkpoint -> lora_b = 0)
        # occupy the bottom half permanently, so this is the path by which they
        # come alive at all.
        #
        # ROTATION vs LEARNING. The run loop now evicts experts as they finish, so
        # by this point most of the batch's experts are gone — and
        # _apply_expert_learning filters tkl_by_expert against loaded_experts
        # (inference.py, `ids = [e for e in tkl_by_expert if e in ...]`). Left
        # alone, rolling residency would silently train ONLY the last few experts
        # to run: the eviction-before-learning bug reappearing through a new door.
        # Reload the active set before the gradient step. Running is cheap to
        # rotate; training is not, so this is where the reload cost is paid.
        active_now = [eo.expert_id for eo in all_expert_outputs]
        to_reload = [e for e in dict.fromkeys(active_now) if e not in self.expert_pool.loaded_experts]
        if to_reload:
            self.expert_pool.load_experts(to_reload)
            absent = [e for e in dict.fromkeys(active_now) if e not in self.expert_pool.loaded_experts]
            if absent:
                # No silent truncation: when k exceeds what the pool can hold,
                # say which experts went untrained rather than reporting a full
                # learning pass that only covered part of the batch.
                print(f"[learn] {len(set(active_now)) - len(absent)}/{len(set(active_now))} experts resident "
                      f"for the gradient step; {len(absent)} could not be reloaded under the RAM cap: {absent}")
        self._apply_expert_learning(tkl_by_expert, fragment_by_expert, central_out.synthesis_hidden, domain)
        # THE GATE LEARNS HERE. Nothing in the system called apply_gate_gradients
        # — every reference to it was inside a comment — so route_head stayed at
        # its random initialisation forever while route_pref was the entire
        # expert ranking. Expert selection was random-but-fixed for the life of
        # the process. This is the step that makes routing adaptive at all.
        self._apply_gate_learning(tokens, r_i_by_expert, composite)
        # CLUSTER FORMATION. spawn_cluster, merge_close_clusters and prune_stale
        # were all unreachable: no cluster could ever be created, merged or
        # removed at runtime, so the cluster set was frozen at whatever the
        # stored pickle held and `lookup` was effectively read-only. A miss
        # produced nothing and the same miss recurred forever.
        self._maybe_spawn_cluster(gate_out, cluster_hit, all_expert_outputs,
                                  tkl_by_expert, r_i_scores, domain)
        # Observe the batch's real memory high-water mark and refresh the usable
        # ceiling from a LIVE reading — other consumers (HF stream buffers) take
        # RAM after boot, so a ceiling fixed at boot drifts out of date silently.
        try:
            from splitter import get_active_memory_mb
            self.diagnostics.set_usable(get_active_memory_mb() + get_available_ram_mb())
            self.diagnostics.observe_memory(len(all_expert_outputs))
        except Exception as e:
            print(f"[warn] memory observation failed: {e}")
        # Persist the latency curves the batch just updated. save_latency_store()
        # existed but nothing on the live path called it, so every measurement
        # from the 994k-token run was lost and load() swallowed the resulting
        # FileNotFoundError in silence.
        try:
            self.convolution.save_latency_store()
        except Exception as e:
            print(f"[warn] latency store save failed: {e}")
        # Apex-nadir refresh fires only when this input is larger than anything
        # calibrated — the one case where ALLOC(T) would be extrapolating.
        self._refresh_apex_nadir(int(total_tokens), tokens, tkl_by_expert, central_out)
        # NOW the experts can go: every consumer that needed them resident has
        # run. Keeping them past this point would only hold RAM that the next
        # batch's selection may want for different experts.
        #
        # But SAVE FIRST. unload_experts() is a plain `del` — it drops the model
        # object, and with it every gradient step _apply_expert_learning just
        # took. save_experts() existed for exactly this and had no caller, so
        # each exchange trained the experts and then discarded the result at the
        # unload. The save writes only trainable_parameters (the LoRA tensors),
        # so it is cheap, and it must sit between the gradient step and the del.
        # Save the RESIDENT set, not `previous_expert_ids` (the last batch's
        # experts). Under rotation those are different sets: an expert that ran
        # early, was evicted, and was reloaded above for its gradient step is not
        # in previous_expert_ids, and saving that list would drop its update.
        trained_now = [e for e in self.expert_pool.loaded_experts]
        if trained_now:
            try:
                self.expert_pool.save_experts(trained_now)
            except Exception as e:
                print(f"[warn] expert checkpoint save failed: {e}")
            self.expert_pool.unload_experts(trained_now)
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

    def _maybe_spawn_cluster(self, gate_out, cluster_hit, expert_outputs,
                             tkl_by_expert: dict, r_i_scores, domain: str):
        """Create a cluster when this input matched none and the batch went well.

        The gate on spawning is spawn_cluster's own: r_i must beat the domain
        mean, so a region of input space only earns a cluster if the pool
        actually handled it better than it handles that domain in general. A miss
        that also went badly teaches nothing and gets no marker.

        Merge and prune run on the same cadence, because spawning without them is
        how the stored set reached 20 clusters that should have been 9.
        """
        if cluster_hit is not None or not expert_outputs:
            return
        r_i_mean = float(sum(r_i_scores) / len(r_i_scores)) if r_i_scores else 0.0
        domain_mean = self.session_tracker.get_domain_mean_r_i(domain)
        # FIRST BATCH IN A NEW DOMAIN. This batch's own activations are already
        # folded into the domain mean, so on the very first batch of an unseen
        # domain the two are equal by construction and spawn_cluster's strict
        # `r_i > domain_mean_r_i` can never pass — the domain would stay
        # permanently clusterless. Nudge the reference down when this batch IS
        # the entire evidence for the domain.
        n_domain = sum(1 for acts in self.session_tracker.activations.values()
                       for a in acts if a.get("domain") == domain)
        if n_domain <= len(r_i_scores):
            domain_mean = 0.0
        ids = [eo.expert_id for eo in expert_outputs]
        try:
            cluster = self.routing_memory.spawn_cluster(
                gate_hidden=gate_out.hidden_states,
                expert_ids=ids,
                tkl_scores=tkl_by_expert,
                r_out_snapshot={},
                l_eff_scores={int(e): float(r_i_scores[i]) for i, e in enumerate(ids)
                              if i < len(r_i_scores)},
                optimal_k=len(ids),
                token_count=int(self.session_tracker.token_count),
                r_i=r_i_mean,
                domain_mean_r_i=domain_mean,
                domain=domain,
            )
        except Exception as e:
            print(f"[warn] cluster spawn failed: {e}")
            return
        if cluster is None:
            return
        before = len(self.routing_memory.clusters)
        try:
            self.routing_memory.merge_close_clusters()
            self.routing_memory.prune_stale(int(self.session_tracker.token_count))
        except Exception as e:
            print(f"[warn] cluster merge/prune failed: {e}")
        after = len(self.routing_memory.clusters)
        print(f"[cluster] spawned {cluster.cluster_id[:8]} in '{domain}' "
              f"(r_i {r_i_mean:.3f} > domain mean {domain_mean:.3f}); "
              f"{before} -> {after} clusters after merge/prune")

    def _grounded_r_i_override(self, input_text: str, expert_outputs, target_ids) -> dict:
        """Leave-one-out delta for a SAMPLE of this batch's experts.

        Returns {expert_id: sigmoid(loss_without - loss_with)} for the sampled
        experts only; the rest keep their cosine score. Sampling is what makes
        this affordable — every expert every batch is k+1 forwards of a 4B model.
        """
        from training import grounded_r_i
        import random as _random
        ids = [eo.expert_id for eo in expert_outputs]
        if not ids:
            return {}
        n = max(1, int(configs.GROUNDED_SAMPLE_K))
        sample = ids if len(ids) <= n else _random.sample(ids, n)
        payload = [{"expert_id": eo.expert_id,
                    "output_text": getattr(eo, "output_text", "") or ""}
                   for eo in expert_outputs]
        try:
            out = grounded_r_i(self.central, input_text, payload, list(target_ids), sample=sample)
        except Exception as e:
            print(f"[warn] grounded r_i failed, keeping cosine scores: {e}")
            return {}
        if out:
            self._grounded_batches = getattr(self, "_grounded_batches", 0) + 1
            lo, hi = min(out.values()), max(out.values())
            print(f"[grounded] batch {self._grounded_batches}: {len(out)} expert(s) scored "
                  f"against real text, r_i in [{lo:.3f}, {hi:.3f}]")
        return out

    def _apply_gate_learning(self, tokens, r_i_by_expert: dict, composite: dict):
        """One L_gate step on the experts that just ran.

            L_eff   route_head preference -> the r_i distribution Central measured
            L_rel   penalise mass landing on experts whose r_i is trending DOWN

        Both targets are measured, not assumed: r_i comes from Central scoring
        real expert output, and staleness is 1 - composite["learning"], where
        learning is 0.5 + 10*slope of that expert's own r_i history. L_dom is
        deliberately absent — it needs a domain label, and feeding the gate's own
        domain output back as the target would train the head toward its own
        random initialisation. It belongs in a labelled training pass.
        """
        from training import apply_gate_gradients
        active = [int(e) for e in r_i_by_expert.keys()]
        if not active or self.gate.net is None:
            return
        # staleness in [0,1]: high = coasting on an r_i that is falling. Experts
        # with too little history have no "learning" component and score 0.0,
        # which makes their L_rel contribution exactly zero rather than a guess.
        staleness = {}
        for eid in active:
            c = composite.get(eid) or {}
            if "learning" in c:
                staleness[eid] = float(max(0.0, min(1.0, 1.0 - float(c["learning"]))))
        lambdas = (self.maml.lambdas[:3] if self.maml is not None
                   else mx.array(configs.LAMBDA_INIT[:3], dtype=mx.float32))
        try:
            stats = apply_gate_gradients(
                self.gate.net, self.gate.optimizer, tokens, lambdas,
                routing_density=None,                       # no label on this path
                active_expert_ids=active,
                l_eff_targets={int(k): float(v) for k, v in r_i_by_expert.items()},
                staleness=staleness,
            )
        except Exception as e:
            print(f"[warn] gate learning step failed: {e}")
            return
        self._gate_steps = getattr(self, "_gate_steps", 0) + 1
        # k=1 is a SILENT no-op and looks healthy: softmax over a single logit is
        # the constant 1.0, so l_eff is exactly -0.0 and its gradient is zero,
        # while l_rel still contributes a non-zero `total` — which means the
        # skip diagnostic below never fires and the log reads like a normal step.
        if len(active) < 2:
            print(f"[gate] step {self._gate_steps}: k={len(active)} — L_eff has no "
                  f"gradient with fewer than two experts (softmax over one logit is "
                  f"constant); only L_rel contributed")
        elif stats.get("total", 0.0) == 0.0:
            # apply_gate_gradients returns all-zero when it REFUSES the step
            # (non-finite loss or gradient). Silence here would look identical to
            # a converged gate, which is how a dead mechanism hides.
            print(f"[gate] step {self._gate_steps} skipped: non-finite loss or gradient")
        elif self._gate_steps % 20 == 0:
            print(f"[gate] step {self._gate_steps}: L={stats['total']:.4f} "
                  f"eff={stats['l_eff']:.4f} rel={stats['l_rel']:.4f}")

    def _apply_expert_learning(self, tkl_by_expert: dict, fragment_by_expert: dict,
                               synthesis_hidden, domain: Optional[str] = None,
                               l_div_weight: Optional[float] = None):
        """EVERY active expert takes a gradient from Central. The bottom tier
        takes spiderweb pressure ON TOP of it.

            all tiers      MSE(expert_hidden, Central synthesis)
            best + middle  + λ_div · peer repulsion   — stay distinct
            rest           + pressure · ‖W − target‖² — be made better

        This previously ran the gradient step ONLY for the bottom tier, because
        it was written as "the spiderweb pass" rather than "the learning pass".
        Healthy experts therefore received no gradient whatsoever — the pool's
        best performers were the only ones frozen, and the only experts that
        learned were the ones being dragged toward the better half.

        Three tiers, each ⌊√k⌋ wide (training.tier_split): best applies pressure
        and receives none, middle is strong-but-underused and is left to its own
        specialisation, rest receives. Attraction and repulsion remain mutually
        exclusive per expert per batch, so the two never cancel."""
        from training import (tier_split, spiderweb_target, spiderweb_pressure,
                              expert_weight_vector, peer_weight_vector,
                              apply_expert_gradients)
        ids = [e for e in tkl_by_expert if e in self.expert_pool.loaded_experts]
        if not ids:
            return
        lam_div = float(l_div_weight if l_div_weight is not None
                        else (self.maml.lambdas[3].item() if self.maml is not None
                              else configs.LAMBDA_INIT[3]))
        # Three-level rank: domain relevance -> domain performance -> overall.
        ranked = self.session_tracker.tkl_rank(
            ids, domain, assigned=self.session_tracker.expert_domains)
        best, middle, rest = tier_split(ranked)
        upper = best + middle
        target = spiderweb_target(
            [expert_weight_vector(self.expert_pool.loaded_experts[e]) for e in best]
        ) if best and rest else None

        def step(eid, **kw):
            model = self.expert_pool.loaded_experts.get(eid)
            frag = fragment_by_expert.get(eid)
            if model is None or frag is None:
                return
            try:
                apply_expert_gradients(model, self.expert_pool.get_optimizer(eid),
                                       frag, synthesis_hidden, check_finite=False, **kw)
            except Exception as e:
                print(f"[warn] expert {eid} gradient step failed: {e}")

        # Upper tiers: Central gradient + peer repulsion, so they keep covering
        # different ground instead of converging on one another.
        for eid in upper:
            peers = [peer_weight_vector(self.expert_pool.loaded_experts[p])
                     for p in upper if p != eid and p in self.expert_pool.loaded_experts]
            peers = [p for p in peers if p is not None]
            step(eid, peer_weights=peers, l_div_weight=lam_div if peers else 0.0)
        # Bottom tier: Central gradient + pressure toward the better half.
        for rank_index, eid in enumerate(reversed(rest)):     # least-bad first
            if target is None:
                step(eid)                                     # no reference: Central only
                continue
            step(eid, web_target=target,
                 web_weight=spiderweb_pressure(rank_index, len(rest),
                                               is_dormant=self.expert_pool.is_dormant(eid)))

    def _refresh_apex_nadir(self, total_tokens: int, tokens: mx.array,
                            tkl_by_expert: dict, central_out):
        """Re-fit ALLOC(T) when an input arrives larger than anything calibrated.
        Inside the calibrated range the regression interpolates and is trusted;
        past it we would be extrapolating, which is exactly what pinned the old
        curves at their floor.

        This ACTIVELY probes. The batch's own records carry a single allocation
        size per expert, and one point cannot determine a curve — so the refresh
        re-runs each selected expert at min / mid / max (probe_sizes(T)) and fits
        from those. Only 2*sqrt(E) experts are probed, chosen as the best and
        worst by live TKL, so the cost is ~2*sqrt(E)*3 forwards rather than a
        full-pool sweep. Hidden states only (generate_text=False) — the score is
        alignment, and generation would triple the cost for nothing."""
        if not self.convolution.needs_refresh(total_tokens):
            return
        try:
            probes = probe_sizes(total_tokens)
            # Rank within the resident set: the sqrt(E) WORST by TKL are the ones
            # nothing routes to, so they are never loaded, and ranking over the
            # whole pool produced a list this method could only discard.
            wanted = self.convolution.refresh_expert_ids(
                tkl_by_expert, resident=self.expert_pool.loaded_experts.keys()
            )
            loaded = [e for e in wanted if e in self.expert_pool.loaded_experts]
            if len(loaded) < len(wanted):
                # No silent truncation: say what was skipped and why.
                print(f"[apex-nadir] refresh at T={total_tokens}: probing "
                      f"{len(loaded)}/{len(wanted)} experts (rest not resident)")
            if len(loaded) < 2:
                return
            for eid in loaded:
                for size in probes.as_tuple():
                    n = max(1, min(int(size), int(tokens.shape[0])))
                    eo = self.expert_pool.expert_forward(
                        eid, tokens[:n], generate_text=False, allocated_tokens=n,
                    )
                    q = self.central.compute_r_i(
                        eo.hidden_states, central_out.contribution_hidden, eo.wall_time,
                        synthesis_hidden=central_out.synthesis_hidden,
                    )
                    # record_probe drops non-finite/None rather than scoring a
                    # failed measurement as 0 — a fake zero would land in the
                    # low-t region the nadir is fitted from.
                    self.convolution.record_probe(eid, int(total_tokens), n, q, wall_time=eo.wall_time)
                    # Feed the COST side too. These three probes are the same
                    # expert measured at three distinct sizes in one pass, which
                    # is exactly the spread that identifies the latency
                    # intercept; record_probe alone files them under the
                    # goldilocks fit and update_latency never sees them, so the
                    # per-expert cost curve had to wait for token counts to vary
                    # across batches instead.
                    self.convolution.update_latency(eid, n, eo.wall_time)
            if self.convolution.close_input(total_tokens) is not None:
                if self.convolution.fit_allocation():
                    # Persist the WHOLE calibration, not just latency. save() is
                    # the only writer of alloc / goldilocks_points /
                    # central_min_cap, and its one caller was run_calibration(),
                    # which nothing on the live path invokes — so a refit that
                    # finally succeeded would have been discarded at exit.
                    try:
                        self.convolution.save()
                        a = self.convolution.alloc
                        print(f"[apex-nadir] ALLOC refit: {math.exp(a.log_a):.3f}*T^{a.b:.3f} "
                              f"from {a.n_points} points, t_max_seen={a.t_max_seen}")
                    except Exception as e:
                        print(f"[warn] calibration save failed: {e}")
        except Exception as e:
            print(f"[warn] apex-nadir refresh failed at T={total_tokens}: {e}")

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
    
    parser = argparse.ArgumentParser(description="Run Dum-E inference test.")
    parser.add_argument("--deployment", action="store_true", help="Run in deployment mode.")
    args = parser.parse_args()
    
    if args.deployment:
        configs.DEPLOYMENT = True
        
    print("Booting Dum-E system...")
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
