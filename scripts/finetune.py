from __future__ import annotations
import argparse
import json
import signal
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import configs
from apex_nadir_convolution import ApexNadirConvolution
from central import CentralModel
from data import authenticate_huggingface, iter_mixture_samples, get_tokenizer
from diagnostics import Diagnostics
from experts import ExpertPool
from gating import GateModel, TripleKSelector, MaskingSchedule, SelectedExpert
from memory import RoutingMemory, SessionTracker
from meta import MAMLOptimiser
from splitter import get_available_ram_mb, prefetch_next_batch
from training import (
    compute_dot_product_peer_gradients,
    apply_gate_gradients,
    apply_expert_gradients,
)
class FinetuneState:
    def __init__(self):
        self.total_tokens = 0
        self.total_batches = 0
        self.total_experts_activated = 0
        self.domain_k_history: Dict[str, List[int]] = defaultdict(list)
        self.loss_history: List[float] = []
        self.r_i_history: List[float] = []
        self.timeline_a_count = 0
        self.timeline_b_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.weight_snapshots: Dict[int, List[mx.array]] = defaultdict(list)
        self.expert_r_i_history: Dict[int, List[mx.array]] = defaultdict(list)
        self.domain_r_i: Dict[str, List[float]] = defaultdict(list)
        self.last_domain_snapshot_tokens: Dict[str, int] = defaultdict(int)
        self.interrupted = False
        self.diagnostics = Diagnostics()
        self.current_x = configs.X_MAX
        if Path("logs/finetune_metrics.json").exists():
            try:
                import json
                with open("logs/finetune_metrics.json", "r") as f:
                    d = json.load(f)
                    self.total_tokens = d.get("total_tokens", 0)
                    self.total_batches = d.get("total_batches", 0)
                    self.timeline_a_count = d.get("timeline_a_count", 0)
                    self.timeline_b_count = d.get("timeline_b_count", 0)
                    self.current_x = d.get("x_next", self.current_x)
            except Exception:
                pass
    def elapsed(self) -> str:
        secs = int(time.time() - self.start_time)
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    def tokens_per_sec(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_tokens / max(elapsed, 1.0)
def append_proof_metric(record: Dict[str, Any], path: str = "logs/proof_metrics.jsonl") -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
def setup_signal_handler(state: FinetuneState):
    def handler(sig, frame):
        print(f"\n[finetune] Interrupted at batch {state.total_batches}, {state.total_tokens} tokens")
        state.interrupted = True
    signal.signal(signal.SIGINT, handler)
def log_progress(
    state: FinetuneState,
    batch_loss: float,
    batch_r_i: float,
    k_used: int,
    domain: str,
    active_expert_ids: List[int],
    timeline_pref: str,
    a_l: int,
    confidence: float,
    thermal_state: float,
    ram_headroom_mb: float,
    x_next: int,
    tokens_per_sec: float,
):
    state.last_log_time = time.time()
    print(
        f"batch={state.total_batches} | "
        f"loss={float(batch_loss):.4f} | "
        f"k={k_used} | "
        f"pref={timeline_pref} | "
        f"a_l={a_l} | "
        f"conf={confidence:.3f} | "
        f"x_next={x_next} | "
        f"thermal={thermal_state:.1f} | "
        f"ram_mb={ram_headroom_mb:.0f} | "
        f"tok/s={tokens_per_sec:.1f} | "
        f"r_i={batch_r_i:.4f} | "
        f"domain={domain} | "
        f"experts_used={active_expert_ids} | "
        f"total_tokens={state.total_tokens}"
    )
def log_stage(batch_number: int, stage: str, **fields: Any):
    extras = " | ".join(f"{key}={value}" for key, value in fields.items())
    if extras:
        print(f"[batch {batch_number}] {stage} | {extras}")
    else:
        print(f"[batch {batch_number}] {stage}")
def save_checkpoint(
    state: FinetuneState,
    convolution: ApexNadirConvolution,
    routing_memory: RoutingMemory,
    maml: MAMLOptimiser,
    gate: Optional[GateModel] = None,
    force: bool = False,
    checkpoint_every_batches: int = 100,
):
    if not force:
        if checkpoint_every_batches <= 0:
            checkpoint_every_batches = 100
        if state.total_batches <= 0 or state.total_batches % checkpoint_every_batches != 0:
            return
    Path("state").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    convolution.save()
    convolution.save_latency_store()
    routing_memory.save(configs.ROUTING_MEMORY_PATH)
    maml.save()
    from mlx.utils import tree_flatten
    import mlx.core as mx
    gate_dir = Path(configs.CHECKPOINT_DIR) / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    if gate is not None:
        mx.save_safetensors(str(gate_dir / "weights.safetensors"), dict(tree_flatten(gate.model.parameters())))
    metrics = {
        "total_tokens": state.total_tokens,
        "total_batches": state.total_batches,
        "elapsed_seconds": int(time.time() - state.start_time),
        "avg_loss_last_100": float(np.mean(state.loss_history[-100:])) if state.loss_history else 0.0,
        "avg_r_i_last_100": float(np.mean(state.r_i_history[-100:])) if state.r_i_history else 0.0,
        "timeline_a_rate": state.timeline_a_count / max(state.timeline_a_count + state.timeline_b_count, 1),
        "timeline_a_count": state.timeline_a_count,
        "timeline_b_count": state.timeline_b_count,
        "tokens_per_sec": state.tokens_per_sec(),
        "x_next": state.current_x,
        "domain_k_means": {d: float(np.mean(ks[-100:])) for d, ks in state.domain_k_history.items() if ks},
        "lambdas": maml.get_lambdas().tolist(),
    }
    with open("logs/finetune_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[checkpoint] Saved at batch {state.total_batches}, {state.total_tokens} tokens")
def classify_domain(text: str) -> str:
    text_lower = text[:500].lower()
    code_signals = ["def ", "class ", "import ", "function ", "return ", "if (", "for (", "```", "print(", "const ", "var ", "let "]
    math_signals = ["theorem", "proof", "equation", "integral", "derivative", "\\frac", "\\sum", "lemma"]
    science_signals = ["abstract", "arxiv", "experiment", "hypothesis", "methodology", "conclusion", "results"]
    code_hits = sum(1 for s in code_signals if s in text_lower)
    math_hits = sum(1 for s in math_signals if s in text_lower)
    science_hits = sum(1 for s in science_signals if s in text_lower)
    if code_hits >= 3:
        return "code"
    if math_hits >= 2:
        return "reasoning"
    if science_hits >= 2:
        return "knowledge"
    return "general"
def run_finetune(
    max_tokens: int = 500_000,
    max_batches: int = 0,
    batch_token_target: int = 256,
    print_every_batches: int = 10,
    checkpoint_every_batches: int = 100,
    seed: int = 42,
    clean: bool = False,
):
    configs.validate_config()
    if clean:
        import shutil
        print("[boot] Cleaning state and logs for a fresh run...")
        if Path("state").exists():
            shutil.rmtree("state")
        if Path("logs/finetune_metrics.json").exists():
            Path("logs/finetune_metrics.json").unlink()
        if Path("logs/proof_metrics.jsonl").exists():
            Path("logs/proof_metrics.jsonl").unlink()
    print("=" * 70)
    print("  STURNUS — Full Fine-Tuning")
    print("=" * 70)
    print(f"  Target tokens:   {max_tokens:,}")
    print(f"  Batch size:      {batch_token_target} tokens")
    print(f"  Datasets:        {', '.join(configs.DATASET_WEIGHTS.keys())}")
    print(f"  Expert pool:     {configs.EXPERT_POOL_SIZE}")
    print("=" * 70)
    authenticate_huggingface()
    print("[boot] HuggingFace auth OK")
    convolution = ApexNadirConvolution(configs.CALIBRATION_PATH, configs.LATENCY_STORE_PATH)
    convolution.load()
    print("[boot] Convolution loaded")
    routing_memory = RoutingMemory()
    routing_memory.load(configs.ROUTING_MEMORY_PATH)
    print("[boot] Routing memory loaded")
    session_tracker = SessionTracker()
    gate = GateModel()
    gate.load()
    print(f"[boot] Gate loaded ({configs.GATE_MODEL_ID})")
    from mlx_lm.tuner.utils import linear_to_lora_layers
    gate.model.freeze()
    lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
    num_layers = len(gate.model.layers) if hasattr(gate.model, "layers") else len(gate.model.model.layers)
    linear_to_lora_layers(gate.model, num_layers, lora_config)
    gate.model.train()
    central = CentralModel()
    # Do NOT call central.load() at boot — the 7B Central model consumes ~4 GB.
    # Loading it here would leave <200 MB for experts. Instead, CentralModel.load()
    # is called lazily on first use (it has an internal _loaded guard).
    print(f"[boot] Central deferred (will load on first use: {configs.CENTRAL_MODEL_ID})")
    expert_pool = ExpertPool(convolution=convolution, session_tracker=session_tracker)
    triple_k = TripleKSelector(convolution=convolution)
    masking = MaskingSchedule()
    gate_optimizer = optim.Adam(learning_rate=configs.LEARNING_RATE)
    expert_optimizers = {
        eid: optim.Adam(learning_rate=configs.LEARNING_RATE)
        for eid in range(configs.EXPERT_POOL_SIZE)
    }
    maml = MAMLOptimiser(gate_model=gate.model)
    maml.load()
    print("[boot] MAML loaded")
    gate_tokenizer = get_tokenizer(configs.GATE_MODEL_ID)
    state = FinetuneState()
    setup_signal_handler(state)
    available_ram = get_available_ram_mb()
    boot_max_concurrent = max(1, int(available_ram // configs.EXPERT_RAM_MB))
    boot_max_concurrent = min(boot_max_concurrent, configs.K_MAX)
    print(f"[boot] Available RAM (before Central load): {available_ram:.0f} MB → max {boot_max_concurrent} concurrent experts")
    print(f"[boot] Starting training loop...")
    print()
    print_every_batches = max(1, print_every_batches)
    record_every_batches = max(1, checkpoint_every_batches)

    # ── prefetch pipeline ────────────────────────────────────────────────────
    # We peek one sample ahead so we can load the next batch's experts in a
    # background thread while the current batch runs forward+backward+save.
    _pf_thread: Optional[threading.Thread] = None
    _pf_event:  Optional[threading.Event]  = None
    _pf_ids:    List[int]                  = []

    def _kick_prefetch(eids: List[int]) -> None:
        nonlocal _pf_thread, _pf_event, _pf_ids
        if not eids or _pf_event is not None:
            return  # already one in-flight
        _pf_event = threading.Event()
        _pf_ids   = list(eids)
        _pf_thread = threading.Thread(
            target=prefetch_next_batch,
            args=(expert_pool, _pf_ids, _pf_event),
            daemon=True,
        )
        _pf_thread.start()

    def _await_prefetch() -> List[int]:
        """Block until the in-flight prefetch finishes; return prefetched IDs."""
        nonlocal _pf_thread, _pf_event, _pf_ids
        if _pf_event is None:
            return []
        _pf_event.wait()
        _pf_thread = _pf_event = None
        ids, _pf_ids = list(_pf_ids), []
        return ids
    # ── end prefetch helpers ─────────────────────────────────────────────────

    data_iter = iter(iter_mixture_samples(seed=seed))
    print("[boot] Waiting for first sample...")
    sample = next(data_iter, None)
    while sample is not None:
        if state.interrupted:
            break
        if max_tokens > 0 and state.total_tokens >= max_tokens:
            break
        if max_batches > 0 and state.total_batches >= max_batches:
            break
        text = sample.text
        if not text or len(text.strip()) < 20:
            sample = next(data_iter, None)
            continue
        domain = classify_domain(text)
        source = sample.source
        token_ids = gate_tokenizer.encode(text)[:configs.MAX_SEQ_LEN]
        if len(token_ids) < configs.FRAGMENT_MIN:
            sample = next(data_iter, None)
            continue
        batch_start_time = time.time()
        tokens = mx.array(token_ids)
        n_tokens = len(token_ids)
        batch_number = state.total_batches + 1
        stage_due = batch_number <= 3
        if stage_due:
            log_stage(batch_number, "sample", source=source, domain=domain, tokens=n_tokens)
        gate_out = gate.forward(tokens)
        k = gate_out.k_per_token
        confidence = gate_out.confidence
        timeline_pref = gate_out.timeline_flag
        a_l = 1 if timeline_pref == "A" else 0
        cluster_hit = routing_memory.lookup(gate_out.hidden_states)
        if stage_due:
            log_stage(batch_number, "gate", pref=timeline_pref, a_l=a_l, conf=f"{confidence:.3f}", k=k, cluster_hit=cluster_hit is not None)
        state.timeline_b_count += 1
        if timeline_pref == "A":
            state.timeline_a_count += 1
        if cluster_hit is not None:
            selected_ids = cluster_hit.top_experts[:k]
            selected = [SelectedExpert(expert_id=eid, distance_to_peak=0.0, domain=domain, is_alpha=False) for eid in selected_ids]
        else:
            selected = triple_k.select_experts(gate_out, session_tracker, masking, state.total_batches)
        if not selected:
            state.total_tokens += n_tokens
            state.total_batches += 1
            continue
        current_ram = get_available_ram_mb()
        x_used = state.current_x
        current_max_concurrent = max(1, int(current_ram // configs.EXPERT_RAM_MB))
        current_max_concurrent = min(current_max_concurrent, configs.K_MAX)
        max_c = min(current_max_concurrent, configs.K_MAX, x_used)
        selected = selected[:max_c]
        requested_ids = [s.expert_id for s in selected]
        if stage_due:
            log_stage(batch_number, "load_start", requested=requested_ids, x_used=x_used, ram_mb=f"{current_ram:.0f}")
        # Wait for any in-flight prefetch, then only load what isn't already in memory
        _await_prefetch()
        ids_to_load = [eid for eid in requested_ids if eid not in expert_pool.loaded_experts]
        try:
            if ids_to_load:
                expert_pool.load_experts(ids_to_load)
        except RuntimeError as e:
            print(f"[warn] Could not load experts {ids_to_load}: {e}")
            expert_pool.unload_experts(requested_ids)
            sample = next(data_iter, None)
            state.total_tokens += n_tokens
            state.total_batches += 1
            continue
        missing = [eid for eid in requested_ids if eid not in expert_pool.loaded_experts]
        selected = [sel for sel in selected if sel.expert_id in expert_pool.loaded_experts]
        expert_ids = [sel.expert_id for sel in selected]
        if not expert_ids:
            state.total_tokens += n_tokens
            state.total_batches += 1
            continue
        if missing:
            print(f"[warn] Skipping unloaded experts: {missing}")
        if stage_due:
            log_stage(batch_number, "load_done", active=expert_ids, missing=missing)
        fragment_size = max(configs.FRAGMENT_MIN, n_tokens // max(len(expert_ids), 1))
        expert_outputs = []
        expert_hidden_states = []
        expert_frag_tokens = []
        for i, sel in enumerate(selected):
            frag_start = i * fragment_size
            frag_end = min(frag_start + fragment_size, n_tokens)
            if frag_start >= n_tokens:
                break
            frag_tokens = tokens[frag_start:frag_end]
            if frag_tokens.shape[0] < configs.FRAGMENT_MIN:
                continue
            eo = expert_pool.expert_forward(sel.expert_id, frag_tokens)
            expert_outputs.append(eo)
            expert_hidden_states.append(eo.hidden_states)
            expert_frag_tokens.append(frag_tokens)
        if not expert_outputs:
            state.total_tokens += n_tokens
            state.total_batches += 1
            continue
        if stage_due:
            log_stage(batch_number, "experts_done", outputs=len(expert_outputs), fragment_size=fragment_size)
        expert_data = [
            {"expert_id": eo.expert_id, "output_text": eo.output_text, "hidden_states": eo.hidden_states, "wall_time": eo.wall_time}
            for eo in expert_outputs
        ]
        central_out = central.forward(text, expert_data, send_to_user=False)
        if stage_due:
            log_stage(batch_number, "central_done", entropy=f"{central_out.reconstruction_entropy:.4f}")
        batch_r_i_scores = []
        expert_r_i_scores: Dict[int, float] = {}
        batch_l_eff_raw = []
        batch_tkl_scores = {}
        for eo in expert_outputs:
            r_i = central.compute_r_i(eo.hidden_states, central_out.contribution_hidden, eo.wall_time)
            r_out = convolution.compute_r_out(eo.expert_id)
            anchor = expert_pool.get_historical_anchor(eo.expert_id)
            tkl = central.compute_tkl(r_i, r_out, anchor, eo.wall_time)
            throughput = eo.token_count / max(eo.wall_time, 1e-6)
            l_eff_raw = r_i + throughput * 0.001
            session_tracker.record_activation(eo.expert_id, eo.token_count, r_i, eo.wall_time, tkl, domain)
            expert_pool.update_domain_score(eo.expert_id, domain, r_i)
            central.update_r_t(eo.expert_id, eo.token_count, eo.wall_time, convolution)
            batch_r_i_scores.append(r_i)
            expert_r_i_scores[eo.expert_id] = r_i
            batch_l_eff_raw.append(l_eff_raw)
            batch_tkl_scores[eo.expert_id] = tkl
            if eo.expert_id not in state.expert_r_i_history:
                state.expert_r_i_history[eo.expert_id] = []
            state.expert_r_i_history[eo.expert_id].append(mx.array(r_i))
            if len(state.expert_r_i_history[eo.expert_id]) > configs.L_REL_N_WINDOWS:
                state.expert_r_i_history[eo.expert_id] = state.expert_r_i_history[eo.expert_id][-configs.L_REL_N_WINDOWS:]
        n_active = len(expert_outputs)
        l_eff_scores = mx.array(batch_l_eff_raw)
        l_eff_sum = mx.sum(mx.abs(l_eff_scores)) + 1e-8
        l_eff_normed = l_eff_scores / l_eff_sum
        selected_mask = mx.ones([n_active])
        domains = ["code", "reasoning", "knowledge", "general"]
        true_domain_idx = domains.index(domain) if domain in domains else 3
        target_list = [0.0] * gate_out.domain_logits.shape[0]
        if true_domain_idx < len(target_list):
            target_list[true_domain_idx] = 1.0
        elif len(target_list) > 0:
            target_list[0] = 1.0
        cluster_counts = mx.array(target_list)
        all_r_i = []
        for eid_list in state.expert_r_i_history.values():
            all_r_i.extend(eid_list[-5:])
        lambdas = maml.get_lambdas()
        all_r_i_list = all_r_i[-configs.L_REL_N_WINDOWS:] if all_r_i else []
        batch_loss = apply_gate_gradients(
            gate_model=gate.model,
            gate_optimizer=gate_optimizer,
            tokens=tokens,
            lambdas=lambdas,
            l_eff_scores=l_eff_normed,
            selected_mask=selected_mask,
            routing_density=cluster_counts,
            r_i_history=all_r_i_list,
            weight_snapshots=expert_hidden_states
        )
        for eo, f_tokens in zip(expert_outputs, expert_frag_tokens):
            if eo.expert_id in expert_pool.loaded_experts:
                expert_model = expert_pool.loaded_experts[eo.expert_id]
                apply_expert_gradients(
                    expert_model=expert_model,
                    expert_optimizer=expert_optimizers[eo.expert_id],
                    tokens=f_tokens,
                    central_synthesis=central_out.synthesis_hidden
                )
        if stage_due:
            log_stage(batch_number, "gradients_done", experts=expert_ids)
        expert_pool.save_experts(expert_ids)

        # ── prefetch next batch while save finishes ───────────────────────────
        # Peek at the next raw sample, skip blanks, run the gate on it, and
        # kick off a background thread to load that batch's experts so they
        # are already in memory by the time the next iteration reaches load_experts.
        next_sample = next(data_iter, None)
        while next_sample is not None and (not next_sample.text or len(next_sample.text.strip()) < 20):
            next_sample = next(data_iter, None)
        if next_sample is not None:
            _nxt_ids = gate_tokenizer.encode(next_sample.text)[:configs.MAX_SEQ_LEN]
            if len(_nxt_ids) >= configs.FRAGMENT_MIN:
                _nxt_tokens  = mx.array(_nxt_ids)
                _nxt_gate    = gate.forward(_nxt_tokens)
                _nxt_ram     = get_available_ram_mb()
                _nxt_max_c   = max(1, min(int(_nxt_ram // configs.EXPERT_RAM_MB), configs.K_MAX, state.current_x))
                _nxt_sel     = triple_k.select_experts(_nxt_gate, session_tracker, masking, state.total_batches + 1)
                _nxt_eids    = [s.expert_id for s in _nxt_sel[:_nxt_max_c]]
                _kick_prefetch(_nxt_eids)  # non-blocking — loads in background
        # ── end prefetch kick ─────────────────────────────────────────────────

        # Unload current experts but keep any that were just prefetched
        expert_pool.unload_experts(expert_ids, keep_buffer=set(_pf_ids))
        if stage_due:
            log_stage(batch_number, "save_unload_done", experts=expert_ids)

        sample = next_sample  # advance the manual iterator
        if len(expert_hidden_states) >= 2:
            peer_loss = compute_dot_product_peer_gradients(expert_hidden_states)
            mx.eval(peer_loss)
        mean_r_i = float(np.mean(batch_r_i_scores)) if batch_r_i_scores else 0.0
        actual_k = len(expert_ids)
        state.loss_history.append(batch_loss)
        state.r_i_history.append(mean_r_i)
        state.domain_k_history[domain].append(actual_k)
        state.total_tokens += n_tokens
        state.total_batches += 1
        state.total_experts_activated += len(expert_ids)
        batch_elapsed = time.time() - batch_start_time
        state.current_x = state.diagnostics.update(state.total_tokens, batch_elapsed, max_c, actual_k)
        latest_diag = state.diagnostics.history[-1]
        batch_tok_s = n_tokens / max(batch_elapsed, 1e-6)
        print_due = state.total_batches <= 10 or state.total_batches % print_every_batches == 0
        record_due = state.total_batches % record_every_batches == 0
        maml.record_k(domain, actual_k, state.total_tokens)
        if maml.should_run_outer_loop(state.total_tokens, maml.state.last_outer_token):
            maml.run_outer_step_from_metrics(
                domain=domain,
                k_value=actual_k,
                reconstruction_entropy=central_out.reconstruction_entropy,
                timeline_a_rate=session_tracker.get_timeline_a_rate(),
                cluster_count=len(routing_memory.clusters),
            )
            maml.state.last_outer_token = state.total_tokens
        migrated_experts = []
        for eo in expert_outputs:
            if expert_pool.check_starvation_eviction(eo.expert_id, domain):
                new_domain = session_tracker.find_migration_target(eo.expert_id, convolution)
                expert_pool.reassign_expert(eo.expert_id, new_domain)
                migrated_experts.append((eo.expert_id, new_domain))
        domain_r_i_history = state.domain_r_i[domain]
        domain_mean_r_i = float(np.mean(domain_r_i_history[-100:])) if domain_r_i_history else 0.0
        state.domain_r_i[domain].append(mean_r_i)
        should_spawn = (
            mean_r_i > domain_mean_r_i
            and cluster_hit is None
        )
        if should_spawn:
            r_out_snap = {eo.expert_id: convolution.compute_r_out(eo.expert_id) for eo in expert_outputs}
            l_eff_snap = {eo.expert_id: float(l_eff_normed[i].item()) for i, eo in enumerate(expert_outputs)}
            routing_memory.spawn_cluster(
                gate_hidden=gate_out.hidden_states, expert_ids=expert_ids,
                tkl_scores=batch_tkl_scores, r_out_snapshot=r_out_snap,
                l_eff_scores=l_eff_snap, optimal_k=actual_k, token_count=state.total_tokens,
                r_i=mean_r_i, domain_mean_r_i=domain_mean_r_i,
            )
        if state.total_batches % 500 == 0 and state.total_batches > 0:
            routing_memory.prune_stale(state.total_tokens)
            routing_memory.merge_close_clusters()
        if print_due:
            if migrated_experts:
                migrated_ids = [eid for eid, _ in migrated_experts]
                print(f"[migration] count={len(migrated_experts)} | experts={migrated_ids} | from={domain}")
            log_progress(
                state=state,
                batch_loss=batch_loss,
                batch_r_i=mean_r_i,
                k_used=actual_k,
                domain=domain,
                active_expert_ids=expert_ids,
                timeline_pref=timeline_pref,
                a_l=a_l,
                confidence=confidence,
                thermal_state=latest_diag.thermal_state,
                ram_headroom_mb=latest_diag.ram_headroom_mb,
                x_next=state.current_x,
                tokens_per_sec=batch_tok_s,
            )
        cluster_count = len(routing_memory.clusters)
        timeline_a_rate = state.timeline_a_count / max(state.timeline_a_count + state.timeline_b_count, 1)
        if record_due:
            append_proof_metric(
                {
                    "record_type": "batch",
                    "time": state.elapsed(),
                    "elapsed_seconds": int(time.time() - state.start_time),
                    "batch": state.total_batches,
                    "tokens": state.total_tokens,
                    "source": source,
                    "domain": domain,
                    "k": int(actual_k),
                    "timeline_pref": timeline_pref,
                    "a_l": int(a_l),
                    "loss": float(batch_loss),
                    "avg_loss": float(np.mean(state.loss_history[-100:])),
                    "r_i": float(mean_r_i),
                    "avg_r_i": float(np.mean(state.r_i_history[-100:])),
                    "confidence": float(confidence),
                    "x_next": int(state.current_x),
                    "thermal": float(latest_diag.thermal_state),
                    "ram_mb": float(latest_diag.ram_headroom_mb),
                    "ssd_read_rate_mb": float(latest_diag.ssd_read_rate_mb),
                    "requested_experts": requested_ids,
                    "active_experts": expert_ids,
                    "expert_r_i": expert_r_i_scores,
                    "cluster_hit": cluster_hit is not None,
                    "cluster_count": cluster_count,
                    "timeline_a_rate": float(timeline_a_rate),
                    "tokens_per_sec": float(batch_tok_s),
                }
            )
            state.last_domain_snapshot_tokens[domain] = state.total_tokens
            append_proof_metric(
                {
                    "record_type": "domain_snapshot",
                    "time": state.elapsed(),
                    "elapsed_seconds": int(time.time() - state.start_time),
                    "batch": state.total_batches,
                    "tokens": state.total_tokens,
                    "domain": domain,
                    "k": int(actual_k),
                    "timeline_pref": timeline_pref,
                    "a_l": int(a_l),
                    "r_i": float(mean_r_i),
                    "x_next": int(state.current_x),
                    "thermal": float(latest_diag.thermal_state),
                    "ram_mb": float(latest_diag.ram_headroom_mb),
                    "cluster_count": cluster_count,
                    "timeline_a_rate": float(timeline_a_rate),
                }
            )
        save_checkpoint(
            state,
            convolution,
            routing_memory,
            maml,
            gate=gate,
            checkpoint_every_batches=record_every_batches,
        )
    print()
    print("=" * 70)
    print("  FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"  Total tokens:     {state.total_tokens:,}")
    print(f"  Total batches:    {state.total_batches:,}")
    print(f"  Elapsed:          {state.elapsed()}")
    print(f"  Tokens/sec:       {state.tokens_per_sec():.1f}")
    print(f"  Final avg loss:   {float(np.mean(state.loss_history[-100:])) if state.loss_history else 0:.4f}")
    print(f"  Final avg R_i:    {float(np.mean(state.r_i_history[-100:])) if state.r_i_history else 0:.4f}")
    print(f"  Timeline A rate:  {state.timeline_a_count / max(state.timeline_a_count + state.timeline_b_count, 1) * 100:.1f}%")
    print(f"  Routing clusters: {len(routing_memory.clusters)}")
    for domain, ks in state.domain_k_history.items():
        if len(ks) > 100:
            early_k = float(np.mean(ks[:50]))
            late_k = float(np.mean(ks[-50:]))
            print(f"  K({domain}):  {early_k:.1f} → {late_k:.1f}  (Δ={late_k - early_k:+.1f})")
    print("=" * 70)
    save_checkpoint(state, convolution, routing_memory, maml, gate=gate, force=True)
    print("[done] Final checkpoint saved")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sturnus full fine-tuning")
    parser.add_argument("--max-tokens", type=int, default=500_000, help="Stop after this many tokens")
    parser.add_argument("--max-batches", type=int, default=0, help="Stop after this many batches (0=unlimited)")
    parser.add_argument("--batch-size", type=int, default=256, help="Target tokens per batch")
    parser.add_argument("--print-every-batches", type=int, default=10, help="Print progress every N batches")
    parser.add_argument("--checkpoint-every-batches", "--checkpoint-interval", dest="checkpoint_every_batches", type=int, default=100, help="Record progress and save checkpoints every N batches")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data sampling")
    parser.add_argument("--clean", action="store_true", help="Start from empty state")
    args = parser.parse_args()
    run_finetune(
        max_tokens=args.max_tokens,
        max_batches=args.max_batches,
        batch_token_target=args.batch_size,
        print_every_batches=args.print_every_batches,
        checkpoint_every_batches=args.checkpoint_every_batches,
        seed=args.seed,
        clean=args.clean,
    )
