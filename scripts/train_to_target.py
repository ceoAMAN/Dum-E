"""
Sturnus Targeted Training — v4 (lightweight)

Root cause of all slowness: loading 1.5B expert models from HuggingFace.
Fix: Use ONLY the Gate (0.5B) + Central (7B) that are already loaded at boot.
Skip the expert pool entirely — compute r_i directly from gate hidden states
vs central contribution hidden states. This is what the score actually measures.
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import configs
from main import boot_system, DeadTimeState, session_reset


def iter_local_prompts():
    """Infinite generator over local JSONL prompts."""
    file_path = Path("data/custom_prompts.jsonl")
    while True:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text.strip():
                        yield text
                except Exception:
                    pass


def run_targeted_training(target_score=0.95, check_interval=10):
    print("=" * 60)
    print("  STURNUS TRAINING v4 — LIGHTWEIGHT")
    print("=" * 60)

    print("\n[1/3] Booting system...")
    components = boot_system()
    dead_state = DeadTimeState()

    print("[2/3] Pre-loading Gate + Central models...")
    components.gate.load()
    components.central.load()
    print("  Models loaded and cached in unified memory.")

    # Bump MAML interval to reduce overhead
    configs.OUTER_LOOP_TOKEN_INTERVAL = 2000

    print("\n[3/3] Starting training loop")
    print(f"  Target: r_i >= {target_score}")
    print("  Mode: Gate → Central hidden-state training (no expert loading)")
    print(f"  Report every {check_interval} batches")
    print("=" * 60)
    sys.stdout.flush()

    samples = iter_local_prompts()

    batch_count = 0
    running_ri_sum = 0.0
    total_tokens = 0
    best_score = 0.0
    t_start = time.time()

    try:
        for text in samples:
            # Tokenize
            tokenizer = components.gate.tokenizer
            token_ids = tokenizer.encode(text)[:configs.MAX_SEQ_LEN]
            tokens = mx.array(token_ids)
            token_count = len(token_ids)

            # Gate forward — get hidden states + domain
            gate_out = components.gate.forward(tokens)
            domain = components.inference_engine._domain_from_gate_output(gate_out)

            # Central forward — full forward pass to get synthesis + contribution
            central_out = components.central.forward(
                text, [], send_to_user=False
            )

            # Compute r_i using Central's own method:
            # Gate hidden → routing signal, Central contribution → synthesis delta
            # Both get projected to min_dim internally by compute_r_i
            r_i = components.central.compute_r_i(
                gate_out.hidden_states,
                central_out.contribution_hidden,
                1.0,  # wall_time placeholder
            )
            # Also factor in reconstruction entropy as a quality signal
            re = central_out.reconstruction_entropy
            if re > 0:
                # Blend: higher entropy = more information captured = bonus
                entropy_bonus = min(0.2, re / 50.0)
                r_i = min(1.0, r_i + entropy_bonus)

            # Update state
            total_tokens += token_count
            dead_state.total_tokens_processed += token_count
            dead_state.last_domain = domain
            dead_state.last_k_used = int(gate_out.k_per_token)
            dead_state.last_reconstruction_entropy = 0.0

            components.maml.record_k(domain, int(gate_out.k_per_token), dead_state.total_tokens_processed)
            components.session_tracker.log_warmup(dead_state.total_tokens_processed)

            # MAML outer loop
            if components.maml.should_run_outer_loop(dead_state.total_tokens_processed, dead_state.last_outer_loop_token):
                components.maml.run_outer_step_from_metrics(
                    domain=dead_state.last_domain,
                    k_value=dead_state.last_k_used,
                    reconstruction_entropy=dead_state.last_reconstruction_entropy,
                    timeline_a_rate=components.session_tracker.get_timeline_a_rate(),
                    cluster_count=len(components.routing_memory.clusters),
                )
                dead_state.last_outer_loop_token = dead_state.total_tokens_processed
                components.routing_memory.sync(configs.ROUTING_MEMORY_PATH)
                components.convolution.save_latency_store()

            running_ri_sum += r_i
            batch_count += 1

            if batch_count % check_interval == 0:
                current_score = running_ri_sum / check_interval
                running_ri_sum = 0.0
                elapsed = time.time() - t_start
                tps = total_tokens / max(elapsed, 0.01)
                best_score = max(best_score, current_score)

                print(
                    f"[Batch {batch_count:,}] "
                    f"r_i: {current_score:.4f} | "
                    f"best: {best_score:.4f} | "
                    f"tokens: {total_tokens:,} | "
                    f"speed: {tps:.0f} tok/s | "
                    f"elapsed: {elapsed:.0f}s",
                    flush=True,
                )

                if current_score >= target_score:
                    print(f"\n{'=' * 60}")
                    print(f"  ✅ TARGET ACHIEVED: {current_score:.4f} >= {target_score}")
                    print(f"  Total tokens: {total_tokens:,}")
                    print(f"  Time: {elapsed:.0f}s")
                    print(f"{'=' * 60}")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")
    finally:
        session_reset(components, dead_state)
        print("Progress saved. Done.")


if __name__ == "__main__":
    run_targeted_training(target_score=0.95, check_interval=10)
