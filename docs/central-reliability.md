# Central reliability — the per-cluster hallucination vector

> Steps 1 and 2 of [build-order.md](build-order.md).

Central is a 4B model synthesising expert output. It hallucinates too, and its
verdict is currently taken at face value.

## Why a vector and not a scalar — DECIDED

The earlier proposal was a scalar hallucination score subtracted from each
expert's contribution.

**A scalar cannot work.** A per-cluster constant subtracted from every candidate
in that cluster is a uniform offset — it cancels in any comparison and changes
nothing about which expert wins. Same result as the routing finding that argmax
is invariant to a pedestal.

A **vector over token types** is differential: it subtracts more where Central is
actually unreliable and less where it is not. That is what makes the subtraction
do work. The vector form fixes the flaw in the scalar form.

## Judge vs instrument

Related and already settled: a cosine `r_i` makes Central the **judge** — its own
bias enters the verdict. A grounded delta makes Central the **instrument**:

```
delta = CE(Central(q), y) - CE(Central(q + expert_text), y)
```

a paired difference, so systematic error cancels. `compute_grounded_r_i` exists
at [central.py](../central.py) and has **zero callers**.

## Blocking dependency — per-token CE

Attributing hallucination to token *types* requires per-token CE.
[training.py:407](../training.py) is (line moved when `per_token` was added):

```python
return nn.losses.cross_entropy(pred, tgt, reduction="none" if per_token else "mean")
```

Mean reduction is precisely what is blind to sparse errors — a handful of badly
wrong tokens vanish into an average over hundreds of fine ones. The `per_token` path now EXISTS and is verified correct, but no caller passes
`per_token=True` yet, so the profile is still uncomputed.

## Where it lives

A field on `VoronoiCluster` ([memory.py:13](../memory.py)), which is losing its
TKL-era fields (`r_out_snapshot`, `l_eff_scores`, `top_experts`, `optimal_k`)
anyway.

## Used as confidence, not only subtraction — PROPOSED

Recommended: the reliability profile weights how much Central's synthesis is
trusted *for this input*, rather than being subtracted from expert scores.
Subtraction is defensible now that it is a vector; confidence is still the
cleaner use. Not yet agreed.

Measure it **held out**, or it Goodharts itself — a reliability score fitted on
the same tokens it is scored against will always look good.

## OPEN — what is a "token type"?

Three readings, materially different builds. Not decided.

1. **Vocabulary class** — numerals, code punctuation, proper nouns, rare tokens.
2. **Position / span** — inside the reasoning chain vs. in the final answer.
3. **Direction in the cluster's own space** — a vector in the same embedding
   space the routing geometry already uses.

I lean **(3)**: it reuses the max-min vectors and cluster basis already built, so
it is one mechanism serving two purposes rather than a second one. Reliability
for a new input becomes a dot product against an accumulated profile — the same
operation as the membership test.
