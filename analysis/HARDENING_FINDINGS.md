# Sturnus — Four-Pillar Hardening (2026-06-24)

Work to make the paper defensible: loss metric, latency, realistic workload, blind eval.
MacBook Air M4, 16 GB. All numbers measured, not projected.

---

## Pillar 1 — Loss metric fix

**Finding (important):** the `avg_loss` recorded by the marathon is NOT a
language-modeling loss — it is the gate's **domain-routing cross-entropy**
(`apply_gate_gradients` returns `λ_dom · l_dom`). It floored at ~0 not because
local_custom is "trivial LM" (the prior explanation) but because the mixture was
skewed toward "general"-classified local_custom (weight 0.46), so the gate
trivially predicted the dominant domain. The metric was hiding, not measuring.

**Fix:**
- De-skewed `DATASET_WEIGHTS`: local_custom 0.46 → 0.10 (renormalizes to ~0.156),
  with an auto-renormalization step so any single weight can be edited without
  hand-balancing the rest. Disabled streams stay 0; sum stays 1.0.
- Added a balanced, hand-written, held-out set `data/heldout_eval.jsonl`
  (10 code / 10 reasoning / 10 knowledge / 10 general) that no training stream
  contains.
- `evaluation.py` + `scripts/eval_heldout.py`: mixture-skew-immune signals —
  gate routing accuracy (gate-only, cheap) and held-out mean r_i + expert MSE
  (full pipeline). Gate routing accuracy now logged to `logs/eval.csv` every
  checkpoint inside the marathon.

**Headline result (honest):** on the balanced held-out set the **current gate
routes at 30% accuracy — barely above the 25% chance baseline.**

| domain | gate routing acc |
|--------|------------------|
| code | 20% |
| reasoning | 70% |
| knowledge | 20% |
| general | 10% |
| **overall** | **30%** |

The old `avg_loss → 0` was the skewed routing-CE collapsing, *not* the gate
learning to route. This is exactly the value the metric fix was meant to expose.
A fresh run on the de-skewed mixture, tracked by `logs/eval.csv`, can now show
whether routing actually improves past chance.

---

## Pillar 2 — Latency / routing overhead

Two redundant passes removed:

| Optimization | Before | After | Win |
|--------------|--------|-------|-----|
| **Gate**: Timeline B ran the backbone twice (`forward()` + `look_ahead()`). Fused into one pass via `forward_with_topography()`. | 43.8 ms | 23.9 ms | **1.83×** |
| **Central**: derived `base_hidden` from the synthesis pass's question-prefix (causal attention ⇒ identical to a separate question-only forward) instead of a second 7B pass. | 794 ms (2 passes) | 587 ms (1 pass) | **−26% / 1.35×** |

- Gate fusion is bit-identical to the old routing decision (verified).
- Removed dead `CentralModel._compute_hidden_mean`.

**Correction to a prior belief:** the journal previously stated the "2× Central
7B passes per batch is architectural and not removable without gutting
self-supervision." That's now partly false — one of the two passes IS removable
(the base reference is a prefix of the synthesis pass). The single remaining 7B
pass is the real floor.

**Honest caveat on "routing overhead < 15%":** the workload harness showed the
routing layer's cost is the *gate forward* (~23 ms), and expert *selection* is
near-free (~0.2 ms). So "routing overhead" as a fraction of end-to-end latency
is dominated by the gate pass; once Central generation runs, routing is already
a small fraction. The lever that matters is the gate pass count (now 1, was 2).

---

## Pillar 3 — Realistic workload (repeated / paraphrased queries)

`scripts/realistic_workload.py`: 50 unique queries × 10 surface paraphrases =
500 prompts, shuffled, fed through the routing layer with the Voronoi cache
spawning on misses.

**Bug found:** `RoutingMemory._recompute_tau` used `VORONOI_ALPHA = 0.30` as the
cold-start threshold (<2 clusters). Measured fingerprint separation:

| | cosine distance |
|---|---|
| same-query paraphrases | mean 0.018, p90 0.033 |
| different queries | mean 0.136, p10 0.063 |

So 0.30 is ~7× too loose — the first cluster swallowed almost everything before
the cache could tighten. **Fix:** bounded tau to the measured separation band
via `VORONOI_TAU_COLD = 0.030` (cold) and `VORONOI_TAU_CEIL = 0.040` (cap on the
warm `ALPHA × mean_dist`).

**Results (after fix):**

| metric | before tau fix | after |
|--------|---------------|-------|
| overall hit rate | 88% | 76% |
| hit rate, 1st half → 2nd half | 84% → 92% | **60% → 93%** (warming) |
| same-query precision (hit matched the right query's cluster) | 47% | **56%** |
| routing latency, hit vs miss | 1.0× | **1.0×** (no speedup) |

**Two honest findings:**
1. The cache **warms** clearly (hit rate 60% → 93% across the run) — repeated
   queries do get cached.
2. The cache gives **no latency benefit** (1.0×) because expert selection was
   never the bottleneck — the gate forward is. Its value is routing
   *consistency*, not speed.

Precision/recall is tunable (`tau ≈ 0.02` → 91% precision / 75% hit; `≈ 0.06` →
69% precision / 86% hit). Default favours precision because a false hit routes to
the wrong experts while a miss just re-runs near-free selection. Remaining
precision ceiling is bounded by the 0.5B gate's mean-pooled fingerprint
resolution + EMA centroid drift.

---

## Pillar 4 — Blind eval (the architectural truth)

`scripts/blind_eval.py`. Reading the code first reframed the question:

- **Central is frozen.** Training updates the gate (routing) and experts
  (hidden-state alignment); Central is never trained/saved. The user-facing
  reply is always base Central.
- **`expert_forward().output_text` is argmax over the INPUT fragment positions**
  — next-token prediction per input token, decoded. As text it is a scrambled
  echo of the question, e.g. for "Write a Python function that checks whether a
  string is a palindrome": `"ing Python program to takes if a given is a
  palindrome. The a Python function that checks..."`. It is a hidden-state
  signal for r_i, not an answer.

So "pipeline vs Central-alone" is identical *by construction* at inference. The
real question the harness tests: **does injecting what the experts produce into
the prompt help or hurt?**

Configs per query: A = `Central.generate(question)` (deployed reply); B =
`Central.generate(question + expert text)`; C = optional external baseline
(`--baseline-model`; Mixtral-8x7B-4bit ≈ 26 GB is infeasible on 16 GB and is
skipped gracefully).

**Result (n=3 demo, code queries):** injecting expert text changed the output
(A≠B in 3/3) but the change is driven by noise; both A and B produced valid
answers because Central is robust and already knew them — the expert text added
no information. On harder queries the noise risks misleading the model.

**Conclusion:** as currently wired, **the MoE expert pipeline is a training-time
apparatus that does not reach the user-facing answer.** The experts/gate matter
for the training objective (r_i, routing), but the deployed reply is base
Central generating from the prompt. To make experts contribute to the *answer*,
the design needs either (a) a meaningful expert summary injected into generation
(not the current argmax echo), or (b) training Central itself. This is the most
important thing to state in the paper's Limitations / Future Work.

Outputs: `logs/blind_eval.jsonl` (all replies + raw expert text per query) and
`logs/blind_eval_judge_prompt.txt` (ready-to-paste GPT-judge prompt).

---

## New artifacts

| file | purpose |
|------|---------|
| `data/heldout_eval.jsonl` | balanced 40-prompt held-out set |
| `evaluation.py` | reusable held-out eval (gate acc, r_i, expert MSE) |
| `scripts/eval_heldout.py` | CLI: `--full` for the heavy eval |
| `scripts/realistic_workload.py` | repeated/paraphrased query cache study (`--full` for end-to-end overhead) |
| `scripts/blind_eval.py` | A/B/baseline quality harness + judge scaffold |
| `logs/eval.csv` | per-checkpoint held-out gate accuracy (mixture-skew-immune) |

## Config changes
- `DATASET_WEIGHTS`: local_custom 0.46 → 0.10 + auto-renormalize.
- `VORONOI_TAU_COLD = 0.030`, `VORONOI_TAU_CEIL = 0.040` (was `VORONOI_ALPHA`
  0.30 fallback).

## Code changes
- `gating.py`: `_backbone`, `_topography_from_hidden`, `forward_with_topography`;
  `forward(tokens, hidden=None)` reuses a precomputed pass.
- `central.py`: single-pass forward (base from synthesis prefix); `_build_input_ids`
  returns `(input_ids, n_question)`; removed dead `_compute_hidden_mean`.
- `inference.py`: `run()` uses the fused gate call; `_timeline_b(topo=...)`.
- `memory.py`: bounded `_recompute_tau`.
- `scripts/finetune.py`: held-out gate accuracy at checkpoint cadence → `logs/eval.csv`.
