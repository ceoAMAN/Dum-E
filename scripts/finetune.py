import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any, Dict
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
from splitter import get_available_ram_mb
from training import (
    apply_gate_gradients,
    apply_expert_gradients,
)
from evaluation import load_eval_set, gate_routing_accuracy

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
):
    configs.validate_config()
    if deployment:
        configs.DEPLOYMENT = True
    if clean:
        import shutil
        if Path("state").exists():
            shutil.rmtree("state")
        if Path("logs/marathon_metrics.json").exists():
            Path("logs/marathon_metrics.json").unlink()

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
    gate.load()
    from mlx_lm.tuner.utils import linear_to_lora_layers
    gate.model.freeze()
    lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
    num_layers = len(gate.model.layers) if hasattr(gate.model, "layers") else len(gate.model.model.layers)
    linear_to_lora_layers(gate.model, num_layers, lora_config)
    gate.model.train()
    print("[boot] Gate loaded and trainable")

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
    current_ram = boot_ram   # cached; refreshed every RAM_REFRESH_BATCHES (vm_stat is a subprocess — too costly per batch)
    RAM_REFRESH_BATCHES = 25
    expert_budget = max(0, boot_ram - 4000)
    hw_max = min(max(1, int(expert_budget / configs.EXPERT_RAM_MB)), configs.K_MAX)
    cache_size = max(2, hw_max)
    expert_pool = ExpertPool(convolution=convolution, session_tracker=session_tracker, max_loaded=cache_size)
    print(f"[boot] RAM={boot_ram:.0f}MB | expert_cap={hw_max} | cache={cache_size}")

    state = MarathonState()

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

    data_iter = iter(iter_mixture_samples(seed=seed))
    sample = next(data_iter, None)

    while sample is not None:
        if state.interrupted or state.total_tokens >= max_tokens:
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

        gate_out = gate.forward(tokens)
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

        fragment_size = max(configs.FRAGMENT_MIN, n_tokens // max(len(expert_ids), 1))
        expert_outputs = []
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
            expert_frag_tokens.append(frag_tokens)

        if not expert_outputs:
            state.total_tokens += n_tokens
            state.total_batches += 1
            sample = next(data_iter, None)
            continue

        expert_data = [{"expert_id": eo.expert_id, "output_text": eo.output_text, "hidden_states": eo.hidden_states, "wall_time": eo.wall_time} for eo in expert_outputs]
        central_out = central.forward(text, expert_data, send_to_user=False)

        batch_r_i_scores = []
        tkl_scores = {}        # expert_id -> tkl, for ranking cluster.top_experts
        r_out_snapshot = {}    # expert_id -> r_out, stored on the cluster

        for eo in expert_outputs:
            r_i = central.compute_r_i(eo.hidden_states, central_out.contribution_hidden, eo.wall_time)
            r_out = convolution.compute_r_out(eo.expert_id)
            anchor = expert_pool.get_historical_anchor(eo.expert_id)
            tkl = central.compute_tkl(r_i, r_out, anchor, eo.wall_time)

            session_tracker.record_activation(eo.expert_id, eo.token_count, r_i, eo.wall_time, tkl, domain)
            expert_pool.update_domain_score(eo.expert_id, domain, r_i)
            central.update_r_t(eo.expert_id, eo.token_count, eo.wall_time, convolution)

            batch_r_i_scores.append(r_i)
            tkl_scores[eo.expert_id] = tkl
            r_out_snapshot[eo.expert_id] = r_out

        # Domain target for l_dom (the only term that trains the gate): one-hot
        # over the 4 domains, matching the 4-dim domain_logits.
        domains = ["code", "reasoning", "knowledge", "general"]
        true_domain_idx = domains.index(domain) if domain in domains else 3
        target_list = [0.0, 0.0, 0.0, 0.0]
        target_list[true_domain_idx] = 1.0
        cluster_counts = mx.array(target_list)

        batch_loss = apply_gate_gradients(
            gate_model=gate.model,
            gate_optimizer=gate_optimizer,
            tokens=tokens,
            lambdas=maml.get_lambdas(),
            routing_density=cluster_counts,
        )
        
        for eo, f_tokens in zip(expert_outputs, expert_frag_tokens):
            if eo.expert_id in expert_pool.loaded_experts:
                expert_model = expert_pool.loaded_experts[eo.expert_id]
                if eo.expert_id not in expert_optimizers:
                    expert_optimizers[eo.expert_id] = optim.Adam(learning_rate=configs.LEARNING_RATE)
                apply_expert_gradients(
                    expert_model=expert_model,
                    expert_optimizer=expert_optimizers[eo.expert_id],
                    tokens=f_tokens,
                    central_synthesis=central_out.synthesis_hidden
                )

        mean_r_i = float(np.mean(batch_r_i_scores)) if batch_r_i_scores else 0.0
        state.loss_history.append(batch_loss)
        state.r_i_history.append(mean_r_i)

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
            if eval_samples:
                try:
                    er = gate_routing_accuracy(gate, eval_samples)
                    pd = er["per_domain"]
                    print(f"[eval] tok={state.total_tokens:,} | gate_acc={er['accuracy']:.1%} | "
                          f"code={pd['code']:.0%} reas={pd['reasoning']:.0%} "
                          f"know={pd['knowledge']:.0%} gen={pd['general']:.0%}", flush=True)
                    with open("logs/eval.csv", "a") as _ef:
                        _ef.write(f"{state.total_batches},{state.total_tokens},{er['accuracy']:.4f},"
                                  f"{pd['code']:.4f},{pd['reasoning']:.4f},"
                                  f"{pd['knowledge']:.4f},{pd['general']:.4f}\n")
                        _ef.flush()
                except Exception as _eval_err:
                    print(f"[warn] held-out eval failed: {type(_eval_err).__name__}: {_eval_err}", flush=True)

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
    )
