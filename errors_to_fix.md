# errors_to_fix.md

Every known defect in Dum-E, verified where verification was possible.
Last updated 2026-09-05.

The codebase is being **rewritten against the new design in [docs/](docs/)**. That
changes what this list is for: a code bug disappears when the file is rewritten, but
an **architectural** defect is reproduced by any rewrite that keeps the same design.
This file is now sorted on that axis.

**Confidence:**

- **[V]** measured by hand or by an agent that reported the numbers. Fact.
- **[3/3]**, **[2/3]** — adversarial verification: N of 3 independent lenses
  (mechanism-trace / counterexample-hunt / predict-signature-then-measure) upheld the
  claim after being instructed to refute it.
- **[R]** AST call-graph reachability. Over-estimates reachability, so "dead" holds
  with margin.
- **[U]** unverified sweep lead.

---

## PART 1 — ARCHITECTURE

A rewrite reproduces every item here unless the **design rule** is adopted.
Full reasoning in [docs/design-rules.md](docs/design-rules.md).

### 1.1 The reward is circular  [3/3 CONFIRMED]

`central.py:157`. r_i — which drives gate training, TKL, cluster spawning, expert
standing and the apex-nadir probes — is the expert's agreement with a Central state
that the expert's own text produced.

Measured signature, predicted before it was looked for:

```
r_i min = 0.511414   across 152 checkpoints / 9,350 batches / 994,231 tokens
```

A hard floor at 0.5, which is exactly what `(cos+1)/2` gives for any
non-anticorrelated pair. The reward **cannot express "this expert hurt"**.

The grounded override is unreachable from `main.py`. AST enumeration of every
`run(...)` call site:

```
main.py:197,226,235,259,310   target_ids: False
inference.py:1045              target_ids: False
scripts/train_grounded.py:87   target_ids: True   <- the only one
```

**Rule:** ground truth REPLACES proxies; it is never blended with them.

### 1.2 Spread-derived weights structurally suppress ground truth  [V]

`central.py:261`, and the same rule in `memory.composite_tkl_pool`. Component weights
are each component's **std across the batch**. Grounded r_i spans 0.202–0.245
(std ≈ 0.015); cosine alignment spans 0.693–0.988 (std ≈ 0.10).

So when both are present the rule down-weights the ground-truth component by **~7×**
in favour of the self-referential one. This is the hollowness mechanism written as
arithmetic — and it would silently defeat the grounded reward if carried over.

**Rule:** never derive a weight from a component's variance. The self-referential
signal always has more.

### 1.3 Expert training target is ungrounded cross-model distillation  [2/3 CONFIRMED]

`training.py:203`. `loss = mse` is the expert's entire per-example objective, against
Central's mean hidden state. No CE, no label, no `y`.

The claim as I originally filed it was **overstated**, and the refuting lens was right:

- The predicted signature `mse -> 0` is **ABSENT**: quintile means 6.56 / 6.54 / 5.41 /
  4.87 / 5.80, final row 4.18, `corr(batch, mse) = -0.28`. It plateaus near 5.
- The self-derived share is only **~14%** — ≤16 expert tokens against a measured mean
  of 106.3 question tokens, mean-pooled unweighted. The target is dominated by
  Central's encoding of the *question*, which is external.

The real defect is worse in a different way: **`EXPERT_D_MODEL=1536` vs
`CENTRAL_D_MODEL=2560`.** `min_dim` truncates, so the MSE compares the first 1536
coordinates of two models' *unaligned* hidden bases. The target is not merely
ungrounded, it is dimensionally arbitrary.

Also: the same `synthesis_hidden` goes to **every** co-active expert
(`inference.py:911`), pulling them toward one shared point — which directly opposes
the peer-repulsion term in the same loss.

**Rule:** never regress one model's hidden state onto another's without a learned map.

### 1.4 Accounting unit != routing unit  [V]

Every expert record is keyed by **domain** — 4 fixed labels from an argmax over the
first 4 dims of a z-scored gate hidden state. Routing is keyed by **cluster** (7–20,
dynamic). Per-(expert, cluster) standing has nowhere to live.

This is the root of three separate items: expert classes can't exist, `chain.py`'s
migration half has no event to observe, and the per-cluster hallucination vector has
no record to write into.

**Rule:** standing is keyed by the routing unit — `(expert_id, cluster_id)` — everywhere.

### 1.5 chain.py is dead for a design reason, not a missing import  [V]

- `MigrationChains.record_move` needs per-expert **cluster** membership.
  `VoronoiCluster.top_experts` is a spawn-time snapshot never updated, and the only
  migration event in the system moves experts between the **4 hardcoded domains**.
- `SizeChains.next_tau` writes a **per-cluster** tau. `RoutingMemory` has a **single
  global** `self.tau` used for every cluster.

So the design produces no cluster-to-cluster transition and has nowhere to put a
per-cluster radius.

**Rule:** name the event that feeds a mechanism and the state it writes, and make both
first-class, before building the mechanism.

### 1.6 R_out's argmax is always the search floor  [V]

`compute_r_out` returns **32.0 for 99/100 experts**. Not a bug — a definition:
`R_out = argmax_t quality(t)/cost(t)`, and with cost fitted **through the origin** the
ratio is monotone decreasing, so its argmax is always the floor. The code's own
docstring derives this at `apex_nadir_convolution.py:531`.

The intercept fix I made does not survive contact with data: all 100 stored latency
curves still have `c0 = 0.0`, because `_fit_latency_coeffs` clamps a negative measured
intercept to zero and refits through the origin.

**Rule:** a per-expert operating point must be fitted from that expert's own
measurements at ≥2 distinct allocations, and must **refuse to return a number** until
it has them.

### 1.7 Calibration fabricates its own inputs  [V]

`data.py:476`:

```
quality_scores      = 1/(1 + 0.01*|tc - 128|)
gradient_coherence  = min(1, tc/64)
wall_times          = tc * 0.001
```

Every "measurement" is a closed-form function of the token count. **No model is run.
`expert_id` is ignored entirely.** The apex curve is fitted from this — which is why
all 100 experts share identical curves.

**Rule:** no curve may be fitted from a signal computed from its own x-axis.

### 1.8 Coldness is a persisted flag, not an evidence count  [V]

`save()` writes arrays for all 100 experts including untouched defaults; `load()` sets
`has_data = True` whenever a key is present. Live: `has_data` True for **all 100**
while **91 have `lat_stats = [0,0,0,0,0]`** — zero real observations. The system is
warm-at-boot from nothing.

**Rule:** coldness is derived from `n_observations`, never from a stored flag. Defaults
are never serialised as if they were measurements.

### 1.9 The thermal sensor is invented and closes a loop on what it governs  [V]

`powermetrics` needs root, so `_read_thermal` falls through to `_estimate_thermal`:
`49 + 34*(0.30*x_scale + ...)` where `x_scale = x_used / X_MAX`. **Running more experts
raises the reported temperature by construction**, and `select_k_cap` then picks fewer.
Concurrency is throttled by a number computed from concurrency.

**Rule:** close a control loop only on a signal measured outside it. If the real sensor
is unavailable, drop the term and log it as unavailable.

### 1.10 Two live metrics named TKL, on incompatible scales  [V]

`compute_tkl` (units tokens²·score/second, floored at 32) and `composite_tkl`
(a weighted mean in [0,1]) are both live and get compared across call sites. Three
distinct bugs are documented in the code as having been caused by this:
`get_masking_rate` ("~0.99 for every expert, churned the hardest workers"),
`check_starvation_eviction` ("0.7 < 16 is always true"), `get_domain_mean_r_i`.

**Rule:** one canonical metric per question, range declared at the definition.

### 1.11 A ranking tier the system writes to itself  [V]

`record_activation` assigns every active expert the **batch's** domain; `tkl_rank` then
reads that label back as its top-priority "domain relevance" tier. It is identically 1
for every candidate — **zero bits**.

**Rule:** a ranking tier must key on evidence written by a *different* event than the
one being ranked.

### 1.12 Membership by side-effect  [R]

`DomainRegistry` implements "membership is EARNED from measured performance" — the rule
`configs.py:244` claims is in force. The running system instead assigns membership as a
side effect of activation. Two owners; the ungrounded one is wired.

**Rule:** membership has exactly one writer, and it is an explicit decision — never a
side effect of having been activated.

### 1.13 Thresholds from different populations combined in one min()  [V]

`memory.py:44`. Mean inter-centroid distance = 0.962, so `VORONOI_ALPHA * mean_dist =
0.289` against `VORONOI_TAU_CEIL = 0.040` — the constant wins by **7×, always**. The
adaptive term is decorative. Cause: `ALPHA*spread` is centroid-to-centroid; the ceiling
was calibrated on a query-to-query population.

**Rule:** never combine two thresholds calibrated on different distributions.

### 1.14 A cap keyed to a clock instead of to the resource  [V]

`cap = max(10, token_count // CLUSTER_CAP_RATE)` where `token_count` is the monotonic
lifetime clock — so the cap only grows and stops binding after ~500 tokens. Meanwhile
`MAX_CLUSTERS_BY_POOL = 10`, derived from the constraining resource, has **zero readers**.

**Rule:** a cap is a function of the resource that constrains it.

### 1.15 Five concurrency caps, no owner  [V]

Not three — **five**, enforced at layers that do not consult each other:
`configs.X_MAX=6` (policy), `splitter.experts_per_batch()` (RAM), `ExpertPool(max_loaded=6)`
(constructor default, never overridden by the RAM formula), `diagnostics.select_k_cap`
(the live governor), `diagnostics.recommended_x` (a second, uncalled governor).

`splitter.py:245` also does `selected_experts[:n_x]` — the **memory** answer silently
overwrites the **routing** answer.

**Rule:** one component owns concurrency; memory schedules the k selected experts over
time and may never reduce k.

### 1.16 Direction drifts online  [V]

`memory.py:63` EMA-drifts the matched centroid on every hit. `chain.py`'s own module
note states the opposite rule. Two live design statements contradict each other and the
code implements both.

**Rule:** direction is immutable after formation; only tau moves.

### 1.17 Cluster geometry has no re-formation event  [V]

Clusters are spawned online from whatever text arrives, then only merged, pruned or
drifted. Nothing calls a rebuild; `chain.reset_for_reformation` has no caller; the
centroids carry no record of the (extractor, tokenizer, gate) that produced them. Any
upstream fix leaves the old geometry in place, permanently and silently.

**Rule:** geometry is derived offline from a stored corpus and stamped with its
producer version; online updates may never create clusters.

### 1.18 Experts receive gapped subsequences  [R]

The live path groups token **indices** by domain across the whole sequence and gathers
`tokens[indices_mx]` — sentences with holes cut through them. The fix exists
(`assign_spans_to_experts`) and is unreachable.

**Rule:** an expert receives a contiguous span; grouping happens in the schedule.

### 1.19 Allocation computed in two places  [R]

`splitter.py:250` calls `compute_r_out` inside the cutting loop, re-deriving an
allocation gating already chose. Two components compute one quantity from different
inputs, so they can never disagree loudly — the splitter just wins.

**Rule:** gating emits `(expert_id, span_length)`; the splitter only cuts.

### 1.20 Three predicates for one lifecycle decision  [V]

`check_stuck_expert` (wired), `check_starvation_eviction` (same question, incompatible
criterion, no action attached), `check_monopoly_overflow` (no remedy exists in the
design at all — allocation is recomputed every batch and nothing can take tokens away).

**Rule:** one lifecycle decision, one criterion, one owner.

### 1.21 No evaluation stage exists  [V]

`expert_weight_std`, `validate_thermal_regression`, `validate_overlap_grads`,
`count_expert_swaps` are dead not because a call was forgotten but because the design
has **no phase whose job is to check that a mechanism is doing anything**. There is an
inference engine and a dead-time orchestrator; neither asks that question.

Grep for validator callers across the whole repo returns only `configs.validate_config`.

**Rule:** every mechanism publishes its effect into one health record that one loop
prints each cycle.

### 1.22 Producer built before consumer  [R]

`compute_central_ce(per_token=True)` is dead for two design reasons at once: it needs
target tokens that the live path structurally lacks, and its intended consumer (the
per-cluster hallucination vector) has no record to live in (see 1.4).

**Rule:** define the consumer first. No metric is implemented until the record it writes
into and the loop that reads it both exist.

---

## PART 2 — REFUTED

Claims from my earlier sweep that did **not** survive verification. Recorded so they
don't get re-filed.

### ALLOC(T) is a closed loop — REFUTED (1 of 3 upheld)

Two parts are true: probe quality **is** self-referential (`compute_r_i` cosine against
a Central state the probed experts produced), and the expert sample **is** score-selected
twice. One part is false: "over a sample the score allocation itself generated" —
`probe_sizes(T)` is a pure function of `T`, referencing nothing from `self.alloc`.

More importantly, **ALLOC(T) is not fitted at all**:

```
alloc = [log_a=0.0, b=1.0, fitted=0.0, t_max_seen=53, n_points=3]
goldilocks = [(24, 8.0), (33, 15.087), (53, 4.0)]
refit: b = -1.0236, span ratio = 2.208
guard requires 0 < b <= 1 AND span >= e   ->  both fail
```

The fit was **rejected by its own admissibility guard**, so `predict()` returns
`float(t)` and `ALLOC(T) = T` identically. The guard worked. That is the one mechanism
in this codebase that correctly refused to produce a number it hadn't earned.

### The anti-hallucination score is manufactured by training — REFUTED

Clause 1 stands: it **is** peer conformity (`consensus = mx.mean(stacked, axis=0)` over
L2-normalised expert states; `no_halluc = 1 - disagreement`).

Clause 2 fails. The conformity is **architectural, not trained**: all 100 experts are
LoRA adapters over **one frozen base**, and an untrained expert has `lora_b = 0` — a
mathematically exact no-op. They start identical.

And it barely matters. Measured on the live tracker:

```
no_halluc: cross-expert std 0.01283  ->  weight 0.0191  (smallest of seven)
           mean 0.9628, saturated at t=0, headroom 0.037
```

At 1.9% of the composite with 3.7% of headroom, it cannot move the ranking.

### Other corrections

- **"20 clusters that should be 9"** — stale. The live file holds **7**, max pairwise
  similarity 0.887 < `SIM_MEMBER` 0.90, so no merge is pending. The *corrupt provenance*
  claim stands (1.17); the arithmetic doesn't.
- **"`get_distance_to_peak` = 1.0 for 100/100"** — it's 0.0 for 99/100 at
  `allocation = 32`, and 1.0 for all at `allocation = 0`. The substantive claim (zero
  variance across the pool, so the term cannot rank) holds either way.
- **"calibration frozen 23 Aug"** — the files are dated 4 Sep. The freeze is real in
  substance (`alloc.fitted = False`, intercept 0.0 for all 100, 91/100 with zero latency
  stats) but the date was wrong.
- **"nine superseded symbols in central.py"** — count unconfirmed. `compute_l_eff` is
  dead; `compute_tkl`, `compute_r_i`, `update_r_t`, `compute_reconstruction_entropy`,
  `compute_grounded_r_i` all still have live callers.

---

## PART 3 — CODE-ONLY

Vanishes when the file is rewritten. Do not spend design time here.

`splitter.py:79` the missing `/2` · `diagnostics.py:507` `recommended_x` (a superseded
duplicate governor) · `splitter.py:356` `validate_overlap_grads` (downstream of an
undecided mechanism) · `splitter.py:328` `count_expert_swaps` (duplicates a number
`select_k_cap` computes inline) · `splitter.py:401` `measure_gate_ram_mb` (a one-line
boot call that was never added — `measure_expert_ram_mb` **is** wired two lines above
where it belongs, `main.py:54`) · `central.py:346` the dead bodies · `diagnostics.py:681`
the pmset one-way ratchet (variable aliasing in one expression: the floor is the previous
output of the same expression) · `gating.py:237` `timeline_flag` · `vectors.py` 7 of 9
functions · `gating.py:331` `Curriculum` + `DomainRegistry` (**but** see 1.12 — they are
dead because a competing ungrounded writer is wired, which is architectural).

---

## PART 4 — STILL OPEN

- **Two critics never ran** — blind-spot hunt and an attack on the *intended* design in
  `docs/`. Both died on the spend limit twice. This is the largest remaining gap: nothing
  has yet attacked the new design.
- **The spec workflow never ran** — 0 of 13 agents. Ownership table, module
  decomposition, persistence contract.
- **Token type** for the reliability vector: vocabulary class vs direction in cluster
  space. Both replacement designs independently recommended **vocabulary class**, against
  what `docs/central-reliability.md` leans toward, on AK grounds: a static `TYPE[V]` array
  is inspectable with `sort | uniq -c`, cannot drift, and cannot fail silently.
- **Solo delta vs leave-one-out.** Identical at k=1, which is 765/765 of the measured run.
  At k>1, LOO scores a correct-but-redundant expert at ~0. Recommendation: solo.
- **Nothing is committed.** ~3,000 lines untracked.

---

## Appendix — fixed 2026-09-04, do not re-file

Verified by running, not reading: gate learns (`route_head` hash `bc7a92e5` ->
`579c73ec`) · ground truth extracted and wired · clusters spawn/merge/prune ·
`SessionTracker` persists · MAML lambdas consumed · `composite` unbound-local crash ·
k=1 zero-gradient silent no-op · `token_count` clock dropped by `save()` (negative ages,
immortal clusters) · unguarded `load()` · first-domain spawn corner · dishonest governor
boot line · mean-reduced CE.

Two I introduced and caught: a local `from splitter import get_available_ram_mb`
shadowing the module-level import, and `Sample.verifiable` typed `bool` when
`extract_pair` returns the answer **string**.
