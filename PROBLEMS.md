# Dum-E — problem register

36 items. Status as of 2026-08-13, after this session's fixes.
Legend: ❌ open · 🟡 partial · 🔧 mechanism done, needs data · ✅ fixed

---

## ❌ OPEN — 11

### Blocker (1 code defect)

| # | Problem | Notes |
|---|---|---|
| 35 | **No trainer.** Nine built mechanisms have no caller | `apply_central_gradients`, `grounded_r_i`, `probe_central_capacity`, `Curriculum`, `assign_pool`, `composite_tkl`, `peer_disagreement`, `experts_per_batch` wiring, `min_cap → Timeline A/B`. Restore `scripts/finetune.py` + `data.py` from `archive/full-stack-1m-benchmarks-2026-07-02` and patch. |

### Resolve themselves on the first real run (3)

| # | Problem |
|---|---|
| 13 | `route_head` statistically identical to fresh init → expert selection is effectively random |
| 14 | The 38 trained experts are mutually independent noise (pairwise \|cos\| < 0.013 over ~5M dims) |
| 17 | Routing memory collapsed: 20 clusters, all `optimal_k=1`, only 5 distinct experts, empty `l_eff_scores` |

### Design decisions (7) — no code until you choose

| # | Question |
|---|---|
| 3 | `DEPLOYMENT` forces Timeline A. Kill it and use `tokens < min_cap → A` instead? |
| 4 | `stream_reply` (voice path) is Central-only by construction. Give it expert context, or keep by design? |
| 5 | Answer hard-capped at 256 tokens, `finish_reason` discarded. Raise, or derive from context window? |
| 10 | `build_geography_batches` splices non-contiguous tokens when Y≥2. Intended, or fix? |
| 11 | Overlap mechanism (`compute_overlap_padding`) is dead code. Wire it or delete it? |
| 21 | "MAML" is `lambdas -= 0.03·targets` — linear, no inner loop, dead second-order path. Rebuild or rename honestly? |
| 28 | Thermal sensor is an invented formula whose `x_scale` term mechanically punishes concurrency. Drop the term or get real telemetry? |

---

## 🟡 PARTIAL — 4

| # | Problem | Where it stands |
|---|---|---|
| 1 | Central is stock at runtime | `CentralModel.save()` + dual objective (model + synthesiser) built and tested. Still stock until the trainer calls them. |
| 6 | Experts get a slice, not the question | Whole **words** now (`snap_cut_to_word`), and the **whole question rides along** in the prompt. The assigned unit is still a slice. |
| 19 | `r_i` truncates across models | Same Qwen lineage + identical vocab (151936). 2560↔1536 still truncates — wants a learned projection. |
| 27 | Memory governor | `_current_x` now from measured RAM (5 on 16GB, was hardcoded 6). The governor itself is still OLS autoregression of its own past choices. |

---

## 🔧 MECHANISM DONE, NEEDS DATA — 2

| # | Problem | Evidence |
|---|---|---|
| 12 | 62 of 100 experts are `lora_b = 0` no-ops | Spiderweb wired into `_timeline_b`, runs every batch. Verified reviving a dead expert 0 → 1296 non-zero params. |
| 18 | Every quality signal is self-referential | `grounded_r_i` verified: **0.827** true / 0.234 irrelevant / **0.187** confidently-wrong — vs cosine's 0.016 total spread. |

---

## ✅ FIXED — 19

### Answer path
| # | Problem | Fix |
|---|---|---|
| 2 | Nadir floor skipped every expert; Timeline B == Timeline A silently | Discard removed from all 3 sites |
| 7 | Experts got a bare string → ChatML model did document continuation | `_build_expert_prompt` uses the tokenizer's own template + role |
| 8 | Invented facts injected as "Expert analyses to consider" | Closed by #7; verified real analysis output |
| 9 | `generation_length()` couldn't accept `total_tokens` → always the 16/32 fallback | Accepts it, prefers `ALLOC(T)` |
| 34 | **bfloat16 crash** in `compute_reconstruction_entropy` — killed every `central.forward()` | Cast inside MLX before numpy |

### Metrics
| # | Problem | Fix |
|---|---|---|
| 20 | K→0 was an algebraic artifact of `L_dom` shrinking the entropy `k` derives from | `k = T / ALLOC(T)` — measurable, falsifiable |
| 22 | `reconstruction_entropy` = softmax over hidden dims, not vocabulary | Real next-token entropy when logits exist |
| 23 | Central judged every expert through a 32-token stub of the question | `context_limit()` = model's real window (262144); question never trimmed |
| 24 | `R_out` algebraically pinned to 32 for all 100 experts, forever | Three-curve convolution (p90 apex / p50 grounding / p10 nadir) ÷ measured cost |

### Training
| # | Problem | Fix |
|---|---|---|
| 15 | `LORA_ALPHA` passed as raw scale — **16× instead of 2×**, 8× hot | `scale = LORA_ALPHA / LORA_R` at all 3 load sites |
| 16 | `save_latency_store()` existed but nothing called it; all measurements lost | Called per batch |

### Machinery
| # | Problem | Fix |
|---|---|---|
| 25 | Three separate hardcoded copies of the domain list | Single source: `gating.DOMAINS` |
| 26 | Masking rate compared r_i ∈[0,1] against mean TKL (floored at 32) → ~0.99 for everyone → best experts churned forever | Compares r_i to peer mean r_i; strong expert now 0.000 |
| 27a | `_current_x = X_MAX` — opened at 6 resident experts before measuring anything | `experts_per_batch()` from `R − √R − central − gate`, halved |
| 29 | Prefetch thread and main loop mutated `loaded_experts` concurrently | `RLock` over both mutators |
| 30 | `EXPERT_GEN_MAX_TOKENS` / `EXPERT_BOOTSTRAP_TOKENS` unreachable dead code | Superseded by `ALLOC` |

### Scalability
| # | Problem | Fix |
|---|---|---|
| 31 | `EXPERT_POOL_SIZE != 100 → raise` — the pool could not grow or shrink | E is a variable; only the impossible is rejected |
| 32 | `[:4]`, `log(4)`, `zeros(4)` — domain count frozen in three places | `len(DOMAINS)` everywhere; `configs.DOMAINS` overrides |
| 33 | `EXPERT_GROUPS` partitioned experts by **index** before seeing data | Unassigned by default; membership earned via curriculum |
| 36 | **`route_head` silently shrank** 120→100 rows on scale-up (`load_weights(strict=False)` replaces rather than skips) | Resize-aware loader: keeps learned rows, new experts at init, logs the resize |

---

## Critical path

```
trainer  →  boot banner  →  smoke run (~50 batches)  →  real run  →  Sunday test
```

The real run alone closes #12, #13, #14, #17.

## Sunday acceptance test

Against a trained checkpoint, **zero code edits**: `EXPERT_POOL_SIZE 100 → 120`, `DOMAINS += ["vision"]`.

| Check | Expected |
|---|---|
| old experts' r_i on old domains | **unchanged** — no degradation from expansion |
| experts 100–119 `lora_b` | non-zero (spiderweb revived them) |
| `vision` membership | ≥ √E/2 = 5, seeded from the middle tier |
| `ALLOC(T)` on a new max-T | refit, `t_max_seen` advanced |
| domain pools | ≥ ceil(√(120/5)) = 5 each |
| code edits required | **0** |

Row 1 is the result. Absorbing a new domain without retraining or degrading the existing experts is what a fixed-topology MoE structurally cannot do.
