import argparse
import json
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator
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
from experts import ExpertPool
from gating import GateModel, TripleKSelector, MaskingSchedule, SelectedExpert
from memory import RoutingMemory, SessionTracker
from meta import MAMLOptimiser
from splitter import (
    get_available_ram_mb,
    measure_expert_ram_mb,
    compute_xy,
    build_geography_batches,
    compute_x_expert_splits,
)
from training import (
    apply_gate_gradients,
    apply_expert_gradients,
    peer_weight_vector,
)
from evaluation import load_eval_set, gate_routing_accuracy
from benchmarks import BenchmarkRecorder

_PREFETCH_SENTINEL = object()

def prefetch(iterable: Iterable, size: int = 8) -> Iterator:
    """Pull from `iterable` on a background thread into a bounded queue so the
    main training loop never blocks on network streaming / _extract_text while
    the GPU is idle. The producer fetches the NEXT sample(s) during the current
    batch's forward/backward passes; the main loop just pops a ready sample.

    Order-preserving (single FIFO producer) so the mixture RNG is consumed in the
    exact same sequence — training is bit-identical, only wall-clock improves.
    Daemon thread: dies with the process on Ctrl-C. A bounded queue (size) caps
    how far ahead it reads so a fast producer can't run away with memory."""
    q: "queue.Queue" = queue.Queue(maxsize=size)

    def producer():
        try:
            for item in iterable:
                q.put(item)
        except Exception as e:
            q.put((_PREFETCH_SENTINEL, e))
            return
        q.put(_PREFETCH_SENTINEL)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = q.get()
        if item is _PREFETCH_SENTINEL:
            return
        if isinstance(item, tuple) and len(item) == 2 and item[0] is _PREFETCH_SENTINEL:
            return  # producer raised; stop cleanly (error already surfaced upstream)
        yield item

class MarathonState:
    def __init__(self):
        self.total_tokens = 0
        self.total_batches = 0
        self.timeline_a_count = 0
        self.timeline_b_count = 0
        self.start_time = time.time()
        self.interrupted = False
        
        self.loss_history = []
        self.r_i_history = []
        self.domain_r_i = {}   # per-domain running mean r_i, gates cluster spawning

        if Path("logs/marathon_metrics.json").exists():
            try:
                with open("logs/marathon_metrics.json") as f:
                    d = json.load(f)
                    self.total_tokens = d.get("total_tokens", 0)
                    self.total_batches = d.get("total_batches", 0)
                    self.timeline_a_count = d.get("timeline_a_count", 0)
                    self.timeline_b_count = d.get("timeline_b_count", 0)
            except Exception:
                pass

    def elapsed(self) -> str:
        secs = int(time.time() - self.start_time)
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def tok_per_sec(self) -> float:
        return self.total_tokens / max(time.time() - self.start_time, 1.0)

def classify_domain(text: str) -> str:
    t = text[:500].lower()
    code = ["def ", "class ", "import ", "function ", "return ", "if (", "for (", "```", "print(", "const ", "var ", "let "]
    math = ["theorem", "proof", "equation", "integral", "derivative", "\\frac", "\\sum", "lemma"]
    science = ["abstract", "arxiv", "experiment", "hypothesis", "methodology", "conclusion"]
    if sum(1 for s in code if s in t) >= 3:
        return "code"
    if sum(1 for s in math if s in t) >= 2:
        return "reasoning"
    if sum(1 for s in science if s in t) >= 2:
        return "knowledge"
    return "general"

def save_checkpoint(state: MarathonState, convolution, routing_memory, maml, gate):
    Path("state").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    convolution.save()
    routing_memory.save(configs.ROUTING_MEMORY_PATH)
    maml.save()
    from mlx.utils import tree_flatten
    gate_dir = Path(configs.CHECKPOINT_DIR) / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(gate_dir / "weights.safetensors"), dict(tree_flatten(gate.model.trainable_parameters())))
    gate.save_route_head()   # persist the learned expert-routing head alongside the LoRA
    
    a_rate = state.timeline_a_count / max(state.timeline_a_count + state.timeline_b_count, 1)
    lam = [float(x) for x in maml.get_lambdas().tolist()]
    metrics = {
        "total_tokens": state.total_tokens,
        "total_batches": state.total_batches,
        "timeline_a_count": state.timeline_a_count,
        "timeline_b_count": state.timeline_b_count,
        "timeline_a_rate": a_rate,
        "tokens_per_sec": state.tok_per_sec(),
        "avg_loss": float(np.mean(state.loss_history[-100:])) if state.loss_history else 0.0,
        "lambdas": {"l_eff": lam[0], "l_dom": lam[1], "l_rel": lam[2], "l_div": lam[3]},
        "avg_r_i": float(np.mean(state.r_i_history[-100:])) if state.r_i_history else 0.0,
    }
    with open("logs/marathon_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Append the lambda trajectory as a time series so the MAML meta-loss
    # DIRECTION can be judged from the real run (does l_dom get suppressed when
    # routing is uncertain? should the targets be negated?). One JSON line per
    # checkpoint — cheap, append-only, easy to plot afterwards.
    with open("logs/lambda_trajectory.jsonl", "a") as f:
        f.write(json.dumps({
            "tokens": state.total_tokens,
            "batches": state.total_batches,
            "timeline_a_rate": a_rate,
            "avg_recon_entropy": float(np.mean(state.r_i_history[-100:])) if state.r_i_history else 0.0,
            "l_eff": lam[0], "l_dom": lam[1], "l_rel": lam[2], "l_div": lam[3],
        }) + "\n")

    # Per-domain K trajectory — the proof the Core Invariant K(D,N) is non-increasing
    # PER DOMAIN, not just in aggregate (the gap raised in review). Each domain's
    # full (token_count, k) history from the MAML K-velocity records, plus the
    # measured velocity (dk/dtoken; negative = K shrinking = invariant holding).
    per_domain_k = {}
    for dom, rec in maml.state.k_velocity_records.items():
        hist = list(rec.k_history)
        per_domain_k[dom] = {
            "history": hist,                      # [(tokens, k), ...]
            "velocity": maml.compute_k_velocity(dom),  # dk/dtoken, None if too short
            "final_k": hist[-1][1] if hist else None,
            "mean_k_recent": float(np.mean([k for _, k in hist[-50:]])) if hist else None,
        }
    with open("logs/per_domain_k.json", "w") as f:
        json.dump({"tokens": state.total_tokens, "domains": per_domain_k}, f, indent=2)

def run_marathon(
    max_tokens: int = 5_000_000,
    batch_token_target: int = 256,
    checkpoint_every: int = 500,
    print_every: int = 50,
    seed: int = 42,
    clean: bool = False,
    deployment: bool = False,
    max_batches: int = 0,
):
    configs.validate_config()
    if deployment:
        configs.DEPLOYMENT = True
    if clean:
        import shutil
        if Path("state").exists():
            shutil.rmtree("state")
        # Wipe per-run append logs so a clean run starts with fresh trajectories
        # (otherwise benchmark/trajectory rows from a prior run bleed into this one).
        for _f in ("marathon_metrics.json", "trajectory.csv", "eval.csv",
                   "benchmarks.csv", "benchmarks_summary.json", "lambda_trajectory.jsonl"):
            p = Path("logs") / _f
            if p.exists():
                p.unlink()
    # Per-batch logging (trajectory.csv) and checkpoints write into these dirs;
    # create them up front so the first print can't crash on a missing directory.
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("state").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  STURNUS — Learning Marathon")
    print("=" * 60)
    print(f"  Target:    {max_tokens:,} tokens")
    print("  Gradients: ON (gate and experts will learn)")
    print("=" * 60)

    authenticate_huggingface()
    print("[boot] HF auth OK")

    convolution = ApexNadirConvolution(configs.CALIBRATION_PATH, configs.LATENCY_STORE_PATH)
    convolution.load()
    routing_memory = RoutingMemory()
    routing_memory.load(configs.ROUTING_MEMORY_PATH)
    session_tracker = SessionTracker()

    gate = GateModel()
    gate.load()   # applies LoRA to the backbone AND builds GateNet (backbone + routing head)
    print("[boot] Gate loaded and trainable (LoRA backbone + learned routing head)")

    central = CentralModel()
    print("[boot] Central deferred")

    triple_k = TripleKSelector(convolution=convolution)
    masking = MaskingSchedule()
    maml = MAMLOptimiser(gate_model=gate.model)
    maml.load()

    gate_optimizer = optim.Adam(learning_rate=configs.LEARNING_RATE)
    expert_optimizers: Dict[int, Any] = {}  # lazy: created on first use per expert

    gate_tokenizer = get_tokenizer(configs.GATE_MODEL_ID)

    boot_ram = get_available_ram_mb()
    measure_expert_ram_mb()   # updates configs.EXPERT_RAM_MB in place to the real per-expert cost
    current_ram = boot_ram   # cached; refreshed every RAM_REFRESH_BATCHES (vm_stat is a subprocess — too costly per batch)
    RAM_REFRESH_BATCHES = 25
    FINITE_CHECK_EVERY = 20   # finite-ness guards force a host sync; amortise them
    # Reserve Central (deferred-loaded, ~CENTRAL_RAM_MB) + GPU headroom for 7B
    # activations / expert generation / allocator, and cap concurrency hard — an
    # over-committed unified memory makes Metal abort the whole run.
    reserve_mb = configs.CENTRAL_RAM_MB + configs.GPU_HEADROOM_MB
    expert_budget = max(0, boot_ram - reserve_mb)
    hw_max = min(max(1, int(expert_budget / configs.EXPERT_RAM_MB)), configs.K_MAX, configs.MAX_CONCURRENT_EXPERTS)
    cache_size = max(1, hw_max)
    expert_pool = ExpertPool(convolution=convolution, session_tracker=session_tracker, max_loaded=cache_size)
    print(f"[boot] RAM={boot_ram:.0f}MB | expert_ram={configs.EXPERT_RAM_MB:.0f}MB(measured) | expert_cap={hw_max} | cache={cache_size}")

    state = MarathonState()
    bench = BenchmarkRecorder(str(configs.LOG_DIR))   # paper benchmark record (losses, A.2.4, observables)

    # Balanced held-out set for the mixture-skew-immune signal. Gate routing
    # accuracy is gate-only (cheap) so we can run it at every checkpoint.
    eval_samples = load_eval_set()
    if eval_samples:
        print(f"[boot] held-out eval set: {len(eval_samples)} balanced prompts")

    def handler(sig, frame):
        print(f"\n[marathon] Interrupted at {state.total_tokens:,} tokens")
        state.interrupted = True
    signal.signal(signal.SIGINT, handler)

    print(f"[boot] Starting from {state.total_tokens:,} tokens\n")

    # Background prefetch: network streaming + _extract_text for the NEXT sample
    # overlap with the current batch's GPU compute instead of stalling it.
    data_iter = iter(prefetch(iter_mixture_samples(seed=seed)))
    sample = next(data_iter, None)

    while sample is not None:
        if state.interrupted or state.total_tokens >= max_tokens:
            break
        if max_batches and state.total_batches >= max_batches:
            break

        text = sample.text
        if not text or len(text.strip()) < 20:
            sample = next(data_iter, None)
            continue

        domain = classify_domain(text)
        token_ids = gate_tokenizer.encode(text)[:configs.MAX_SEQ_LEN]
        if len(token_ids) < configs.FRAGMENT_MIN:
            sample = next(data_iter, None)
            continue

        t0 = time.time()
        tokens = mx.array(token_ids)
        n_tokens = len(token_ids)

        # Geography-first look-ahead: ONE gate backbone pass yields both the routing
        # output and the full-prompt domain topography used to build homogeneous
        # expert batches below.
        gate_out, topo = gate.forward_with_topography(tokens)
        k = gate_out.k_per_token
        confidence = gate_out.confidence
        timeline = gate_out.timeline_flag

        if deployment or configs.DEPLOYMENT:
            timeline = "A"
        else:
            # During training always explore timeline B so the gate receives
            # gradient signal from l_dom. It learns routing via the supervision
            # from classify_domain(text); we only trust its own confidence for
            # inference after training is complete.
            timeline = "B"
            k = configs.K_DEFAULT

        if timeline == "A":
            state.timeline_a_count += 1
        else:
            state.timeline_b_count += 1

        if timeline == "A":
            central.forward(text, [], send_to_user=False)
            state.total_tokens += n_tokens
            state.total_batches += 1
            elapsed = time.time() - t0
            maml.record_k(domain, 0, state.total_tokens)
            session_tracker.record_timeline_a(n_tokens)
            if state.total_batches % print_every == 0:
                a_rate = state.timeline_a_count / max(state.timeline_a_count + state.timeline_b_count, 1)
                print(
                    f"[A] batch={state.total_batches} | "
                    f"tok={state.total_tokens:,} | "
                    f"tok/s={n_tokens/max(elapsed,1e-6):.0f} | "
                    f"conf={confidence:.3f} | "
                    f"A_rate={a_rate:.1%} | "
                    f"{state.elapsed()}"
                )
            if state.total_batches % checkpoint_every == 0:
                save_checkpoint(state, convolution, routing_memory, maml, gate)
            sample = next(data_iter, None)
            continue

        k = max(1, k)

        cluster_hit = routing_memory.lookup(gate_out.hidden_states)
        if cluster_hit is not None:
            selected_ids = cluster_hit.top_experts[:k]
            selected = [SelectedExpert(expert_id=eid, distance_to_peak=0.0, domain=domain, is_alpha=False) for eid in selected_ids]
        else:
            loaded_set = set(expert_pool.loaded_experts.keys())
            selected = triple_k.select_experts(gate_out, session_tracker, masking, state.total_batches, loaded_set)

        if not selected:
            state.total_tokens += n_tokens
            state.total_batches += 1
            sample = next(data_iter, None)
            continue

        if state.total_batches % RAM_REFRESH_BATCHES == 0:
            current_ram = get_available_ram_mb()   # refresh occasionally, not every batch
        max_c = min(max(1, int(current_ram // configs.EXPERT_RAM_MB)), configs.K_MAX, hw_max)
        selected = selected[:max_c]
        requested_ids = [s.expert_id for s in selected]

        ids_to_load = [eid for eid in requested_ids if eid not in expert_pool.loaded_experts]
        try:
            if ids_to_load:
                expert_pool.load_experts(ids_to_load)
        except RuntimeError:
            expert_pool.unload_experts(requested_ids)
            sample = next(data_iter, None)
            state.total_tokens += n_tokens
            state.total_batches += 1
            continue

        selected = [s for s in selected if s.expert_id in expert_pool.loaded_experts]
        expert_ids = [s.expert_id for s in selected]
        if not expert_ids:
            state.total_tokens += n_tokens
            state.total_batches += 1
            sample = next(data_iter, None)
            continue

        # ── X/Y GEOGRAPHY ───────────────────────────────────────────────────
        # Split the sequence into domain-homogeneous batches (from the look-ahead
        # topography), then divide each batch across the selected experts by their
        # R_out share. X = concurrent experts, Y = cycles; OOM-proof by construction
        # (Y scales time, never RAM). Each expert hands Central BOTH channels —
        # hidden states + a REAL generated analysis (length governed by R_out, else
        # the safety valve). Input fragment size is the R_out share computed here.
        r_out_mean = max(1.0, convolution.compute_r_out_mean(expert_ids))
        geometry = compute_xy(
            max(1, n_tokens), r_out_mean,
            max(current_ram, float(configs.EXPERT_RAM_MB)),
            x_override=len(expert_ids),
        )
        geo_batches = build_geography_batches(tokens, topo.domain_map, geometry.Y)
        expert_outputs = []
        expert_frag_tokens = []
        for gb in geo_batches:
            if len(gb.token_indices) == 0:
                continue
            for frag in compute_x_expert_splits(gb, selected, geometry.X, convolution):
                if frag.below_nadir or frag.expert_id not in expert_pool.loaded_experts:
                    continue
                if frag.tokens.shape[0] < configs.FRAGMENT_MIN:
                    continue
                eo = expert_pool.expert_forward(
                    frag.expert_id, frag.tokens, generate_text=True,
                    max_tokens=convolution.generation_length(frag.expert_id),
                )
                expert_outputs.append(eo)
                expert_frag_tokens.append(frag.tokens)

        if not expert_outputs:
            state.total_tokens += n_tokens
            state.total_batches += 1
            sample = next(data_iter, None)
            continue

        expert_data = [{"expert_id": eo.expert_id, "output_text": eo.output_text, "hidden_states": eo.hidden_states, "wall_time": eo.wall_time} for eo in expert_outputs]
        central_out = central.forward(text, expert_data, send_to_user=False)

        lambdas = maml.get_lambdas()
        check_finite = (state.total_batches % FINITE_CHECK_EVERY == 0)
        active_ids = [eo.expert_id for eo in expert_outputs]

        batch_r_i_scores = []
        tkl_scores = {}        # expert_id -> tkl, ranks cluster.top_experts
        r_out_snapshot = {}    # expert_id -> r_out, stored on the cluster
        l_eff_targets = {}     # expert_id -> efficiency score, trains the gate routing head (L_eff)
        staleness = {}         # expert_id -> [0,1], 1 = coasting on stale reputation (L_rel)

        for eo in expert_outputs:
            # Dual R_i: direction (vs contribution) + compatibility (vs synthesis).
            r_i = central.compute_r_i(
                eo.hidden_states, central_out.contribution_hidden, eo.wall_time,
                synthesis_hidden=central_out.synthesis_hidden,
            )
            r_out = convolution.compute_r_out(eo.expert_id)
            anchor = expert_pool.get_historical_anchor(eo.expert_id)
            tkl = central.compute_tkl(r_i, r_out, anchor, eo.wall_time)
            l_eff_targets[eo.expert_id] = central.compute_l_eff(
                eo.hidden_states, central_out.synthesis_hidden, eo.token_count, eo.wall_time
            )
            # Staleness = how far this batch's r_i fell below the expert's running
            # reputation (EMA read BEFORE this batch updates it).
            ema = expert_pool.domain_scores[eo.expert_id].get(domain, 0.0)
            staleness[eo.expert_id] = max(0.0, min(1.0, 1.0 - r_i / (ema + 1e-6))) if ema > 1e-6 else 0.0

            session_tracker.record_activation(eo.expert_id, eo.token_count, r_i, eo.wall_time, tkl, domain)
            expert_pool.update_domain_score(eo.expert_id, domain, r_i)
            central.update_r_t(eo.expert_id, eo.token_count, eo.wall_time, convolution)

            batch_r_i_scores.append(r_i)
            tkl_scores[eo.expert_id] = tkl
            r_out_snapshot[eo.expert_id] = r_out

        # Domain target for L_dom: one-hot over the 4 domains.
        domains = ["code", "reasoning", "knowledge", "general"]
        true_domain_idx = domains.index(domain) if domain in domains else 3
        target_list = [0.0, 0.0, 0.0, 0.0]
        target_list[true_domain_idx] = 1.0
        cluster_counts = mx.array(target_list)

        # L_gate = λ_eff·L_eff + λ_dom·L_dom + λ_rel·L_rel — trained through the gate's
        # LoRA backbone AND its learned routing head (GateNet), in one step. Returns
        # the per-term breakdown for the benchmark record.
        gate_losses = apply_gate_gradients(
            gate_net=gate.net,
            gate_optimizer=gate_optimizer,
            tokens=tokens,
            lambdas=lambdas,
            routing_density=cluster_counts,
            active_expert_ids=active_ids,
            l_eff_targets=l_eff_targets,
            staleness=staleness,
            check_finite=check_finite,
        )

        # Peer weight vectors snapshotted BEFORE any expert update, so every expert
        # repels the same fixed peer set this batch (L_div repulsion, weight λ_div).
        peer_vecs = {eid: peer_weight_vector(expert_pool.loaded_experts[eid])
                     for eid in active_ids if eid in expert_pool.loaded_experts}
        l_div_weight = float(lambdas[3])

        expert_mse_scores = []
        expert_div_sims = []
        for eo, f_tokens in zip(expert_outputs, expert_frag_tokens):
            if eo.expert_id in expert_pool.loaded_experts:
                expert_model = expert_pool.loaded_experts[eo.expert_id]
                if eo.expert_id not in expert_optimizers:
                    expert_optimizers[eo.expert_id] = optim.Adam(learning_rate=configs.LEARNING_RATE)
                peers = [v for j, v in peer_vecs.items() if j != eo.expert_id and v is not None]
                el = apply_expert_gradients(
                    expert_model=expert_model,
                    expert_optimizer=expert_optimizers[eo.expert_id],
                    tokens=f_tokens,
                    central_synthesis=central_out.synthesis_hidden,
                    peer_weights=peers,
                    l_div_weight=l_div_weight,
                    check_finite=check_finite,
                )
                expert_mse_scores.append(el["mse"])
                expert_div_sims.append(el["l_div_sim"])

        mean_r_i = float(np.mean(batch_r_i_scores)) if batch_r_i_scores else 0.0
        state.loss_history.append(gate_losses["total"])
        state.r_i_history.append(mean_r_i)

        # ── BENCHMARKS ── every loss term, the audit-A.2.4 expert→reply signal
        # (contribution_norm = ‖synthesis - base‖, the delta experts cause in Central,
        # + r_i), and the acceptance observables. All from the real run.
        bench.note_cluster_lookup(cluster_hit is not None)
        bench.record_batch({
            "gate_total": gate_losses["total"], "l_eff": gate_losses["l_eff"],
            "l_dom": gate_losses["l_dom"], "l_rel": gate_losses["l_rel"],
            "expert_mse": float(np.mean(expert_mse_scores)) if expert_mse_scores else 0.0,
            "l_div_sim": float(np.mean(expert_div_sims)) if expert_div_sims else 0.0,
            "r_i": mean_r_i,
            "contribution_norm": float(mx.linalg.norm(central_out.contribution_hidden).item()),
            "k": float(len(active_ids)),
            "recon_entropy": float(central_out.reconstruction_entropy),
        })

        # Routing memory: when this batch routed BETTER than the domain's running
        # average (good route) and we didn't already reuse a cached one, save it as
        # a Voronoi cluster. Future prompts with a similar gate fingerprint hit the
        # cache (routing_memory.lookup) and skip expert re-selection — routing gets
        # faster and more consistent as training proceeds. spawn_cluster internally
        # no-ops if r_i <= domain_mean_r_i, so it only keeps the winners.
        prev_domain_mean = state.domain_r_i.get(domain, 0.0)
        if cluster_hit is None and expert_ids:
            routing_memory.spawn_cluster(
                gate_hidden=gate_out.hidden_states,
                expert_ids=expert_ids,
                tkl_scores=tkl_scores,
                r_out_snapshot=r_out_snapshot,
                l_eff_scores={},
                optimal_k=len(expert_ids),
                token_count=state.total_tokens,
                r_i=mean_r_i,
                domain_mean_r_i=prev_domain_mean,
                domain=domain,
            )
        # Update the per-domain r_i EMA used as the spawn threshold next time.
        state.domain_r_i[domain] = (
            0.9 * prev_domain_mean + 0.1 * mean_r_i if domain in state.domain_r_i else mean_r_i
        )

        state.total_tokens += n_tokens
        state.total_batches += 1
        elapsed = time.time() - t0

        # ── OBSERVABILITY ── runs UNCONDITIONALLY right after the counter bumps,
        # so logging/checkpointing can never be skipped by a failure in the
        # adaptive block below (the bug that made [learn]/[ckpt] never fire).
        if state.total_batches % print_every == 0:
            lam = [float(x) for x in maml.get_lambdas().tolist()]
            avg_loss = float(np.mean(state.loss_history[-print_every:]))
            avg_r_i = float(np.mean(state.r_i_history[-print_every:]))
            print(
                f"[learn] batch={state.total_batches} | "
                f"tok={state.total_tokens:,} | "
                f"k={len(expert_ids)} | "
                f"loss={avg_loss:.4f} | "
                f"r_i={avg_r_i:.4f} | "
                f"λ=[eff{lam[0]:.2f} dom{lam[1]:.2f} rel{lam[2]:.2f} div{lam[3]:.2f}] | "
                f"tok/s={n_tokens/max(elapsed,1e-6):.0f} | "
                f"experts={expert_ids} | "
                f"{state.elapsed()}",
                flush=True,
            )
            with open("logs/trajectory.csv", "a") as _tf:
                _tf.write(f"{state.total_batches},{state.total_tokens},{avg_loss:.6f},"
                          f"{avg_r_i:.6f},{len(expert_ids)},{len(routing_memory.clusters)},"
                          f"{lam[0]:.6f},{lam[1]:.6f},{lam[2]:.6f},{lam[3]:.6f}\n")
                _tf.flush()

        if state.total_batches % checkpoint_every == 0:
            routing_memory.merge_close_clusters()        # fold near-duplicate routes
            routing_memory.prune_stale(state.total_tokens)  # drop cold low-confidence ones
            save_checkpoint(state, convolution, routing_memory, maml, gate)
            expert_pool.save_experts()   # persist all loaded experts at checkpoint cadence
            print(f"[ckpt] batch={state.total_batches} | tok={state.total_tokens:,} | clusters={len(routing_memory.clusters)}", flush=True)
            # Mixture-skew-immune signal: gate routing accuracy on the balanced
            # held-out set. Unlike avg_loss (gate CE on the skewed mixture) this
            # number is comparable across runs and actually reflects learning.
            gate_acc = 0.0
            if eval_samples:
                try:
                    er = gate_routing_accuracy(gate, eval_samples)
                    gate_acc = er["accuracy"]
                    pd = er["per_domain"]
                    print(f"[eval] tok={state.total_tokens:,} | gate_acc={gate_acc:.1%} | "
                          f"code={pd['code']:.0%} reas={pd['reasoning']:.0%} "
                          f"know={pd['knowledge']:.0%} gen={pd['general']:.0%}", flush=True)
                    with open("logs/eval.csv", "a") as _ef:
                        _ef.write(f"{state.total_batches},{state.total_tokens},{gate_acc:.4f},"
                                  f"{pd['code']:.4f},{pd['reasoning']:.4f},"
                                  f"{pd['knowledge']:.4f},{pd['general']:.4f}\n")
                        _ef.flush()
                except Exception as _eval_err:
                    print(f"[warn] held-out eval failed: {type(_eval_err).__name__}: {_eval_err}", flush=True)
            # Consolidated paper-benchmark row: all loss terms, the A.2.4 expert→reply
            # signal, and the manual's acceptance observables, in one place.
            try:
                k_vels = [v for v in (maml.compute_k_velocity(d) for d in maml.state.k_velocity_records) if v is not None]
                brow = bench.checkpoint(
                    batch=state.total_batches,
                    tokens=state.total_tokens,
                    cluster_count=len(routing_memory.clusters),
                    timeline_a_rate=session_tracker.get_timeline_a_rate(),
                    gate_acc=gate_acc,
                    expert_weight_std=expert_pool.expert_weight_std(list(expert_pool.loaded_experts.keys())),
                    k_velocity_mean=float(np.mean(k_vels)) if k_vels else 0.0,
                )
                print(f"[bench] L=[eff{brow['l_eff']:.3f} dom{brow['l_dom']:.3f} rel{brow['l_rel']:.3f}] "
                      f"mse={brow['expert_mse']:.3f} div_sim={brow['l_div_sim']:.3f} | "
                      f"r_i={brow['r_i']:.3f} contrib={brow['contribution_norm']:.3f} | "
                      f"wstd={brow['expert_weight_std']:.4f} hit={brow['cluster_hit_rate']:.0%} "
                      f"kvel={brow['k_velocity_mean']:.2e}", flush=True)
            except Exception as _bench_err:
                print(f"[warn] benchmark record failed: {type(_bench_err).__name__}: {_bench_err}", flush=True)

        # ── ADAPTIVE ── MAML meta-update + expert migration. Wrapped so any failure
        # here is surfaced as a [warn] instead of silently aborting the rest of the
        # loop body (which previously killed all logging/checkpointing downstream).
        try:
            maml.record_k(domain, len(expert_ids), state.total_tokens)
            if maml.should_run_outer_loop(state.total_tokens, maml.state.last_outer_token):
                maml.run_outer_step_from_metrics(
                    domain=domain,
                    k_value=len(expert_ids),
                    reconstruction_entropy=central_out.reconstruction_entropy,
                    timeline_a_rate=session_tracker.get_timeline_a_rate(),
                    cluster_count=len(routing_memory.clusters),
                )
                maml.state.last_outer_token = state.total_tokens
            for eo in expert_outputs:
                if expert_pool.check_starvation_eviction(eo.expert_id, domain):
                    new_domain = session_tracker.find_migration_target(eo.expert_id, convolution)
                    expert_pool.reassign_expert(eo.expert_id, new_domain)
        except Exception as _adapt_err:
            if state.total_batches % print_every == 0:
                print(f"[warn] adaptive step failed: {type(_adapt_err).__name__}: {_adapt_err}", flush=True)

        sample = next(data_iter, None)

    print()
    print("=" * 60)
    print("  MARATHON COMPLETE")
    print("=" * 60)
    print(f"  Tokens:    {state.total_tokens:,}")
    print(f"  Batches:   {state.total_batches:,}")
    print(f"  Elapsed:   {state.elapsed()}")
    print(f"  Tok/sec:   {state.tok_per_sec():.1f}")
    if state.loss_history:
        print(f"  Final Loss:{float(np.mean(state.loss_history[-100:])):.4f}")
    print(f"  Clusters:  {len(routing_memory.clusters)}")
    print("=" * 60)
    save_checkpoint(state, convolution, routing_memory, maml, gate)
    expert_pool.save_experts()   # final persist (also covers Ctrl-C interrupt)
    print("[done] Final checkpoint saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--checkpoint-every-batches", type=int, default=None)
    parser.add_argument("--print-every-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--deployment", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0, help="Stop after N batches (0 = unlimited). For smoke tests.")
    args = parser.parse_args()
    ckpt_every = args.checkpoint_every_batches or args.checkpoint_every
    run_marathon(
        max_tokens=args.max_tokens,
        batch_token_target=args.batch_size,
        checkpoint_every=ckpt_every,
        print_every=args.print_every_batches,
        seed=args.seed,
        clean=args.clean,
        deployment=args.deployment,
        max_batches=args.max_batches,
    )
