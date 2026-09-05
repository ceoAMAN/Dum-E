# Design rules for the rewrite

Twenty-two rules, each derived from a **verified** defect in the old system. The rule
is the part that matters: a rewrite that adopts the rule cannot reproduce the defect.
Evidence for each is in [../errors_to_fix.md](../errors_to_fix.md) Part 1.

## Grounding

1. **Ground truth REPLACES proxies.** It is never blended with them.
2. **Never derive a weight from a component's variance.** The self-referential signal
   always has more spread, so a spread rule mechanically suppresses ground truth —
   measured at ~7× in the old `compute_r_i_batch`.
3. **No curve may be fitted from a signal computed from its own x-axis.**
4. **Close a control loop only on a signal measured outside it.** If the real sensor is
   unavailable, drop the term with weight 0 and log it as unavailable — never model it
   from the control variable.
5. **Never regress one model's hidden state onto another's** without a learned map.
   Different bases; truncating to `min_dim` compares arbitrary coordinates.

## Ownership

6. **One component owns concurrency.** Every other layer queries it; none clamps,
   defaults, or re-derives a bound. Memory schedules the k selected experts over time
   and may never reduce k.
7. **Membership has exactly one writer,** and it is an explicit decision — never a side
   effect of having been activated.
8. **One canonical metric per question,** with its range declared at the definition. A
   superseded metric is deleted, not retained. No comparison crosses two differently
   named metrics.
9. **One lifecycle decision, one criterion, one owner.** Every predicate names the
   single action it triggers and the single loop that runs it.
10. **Gating emits `(expert_id, span_length)`.** The splitter only cuts at the offsets
    it is given and may not import the allocation model.

## Keying

11. **Standing is keyed by the routing unit** — `(expert_id, cluster_id)` — everywhere
    it is written and read. If a label is not what routing selects on, it may not be
    what scoring aggregates on.
12. **A ranking tier must key on evidence written by a different event** than the one
    being ranked. A label the system assigns itself at activation time is not a signal.
13. **A cap is a function of the resource that constrains it,** never of a monotonic
    clock. The derived bound must be the one the enforcement path reads.
14. **Never combine two thresholds calibrated on different populations** in one
    `min`/`max`. A threshold is validated against the exact distribution it applies to.

## Geometry

15. **A cluster's direction is immutable after formation.** All online adaptation
    happens in the radius (`tau`) only; the centroid array is written exactly once.
16. **Geometry is derived offline from a stored corpus** and stamped with the
    `(extractor, tokenizer, gate)` version that produced it. Any change to those
    invalidates and rebuilds it. Online updates may never create clusters.
17. **An expert receives a contiguous span** of the original token array. Grouping by
    domain or expert happens in the schedule — the order of processing — never by
    gathering scattered token indices.

## Construction

18. **Name the event and the state first.** Before building an adaptive mechanism, name
    the event that feeds it and the state it writes, and make both first-class in the
    live path. `chain.py` is dead because neither existed.
19. **Define the consumer before the producer.** No metric is implemented until the
    record it writes into and the loop that reads it both exist. And no metric requiring
    ground truth may sit on a path that structurally has none.
20. **Refuse rather than fall back.** A per-expert quantity must be fitted from that
    expert's own measurements at ≥2 distinct points and must refuse to return a number
    until it has them. A scalar with zero variance across the pool must raise, never rank.
21. **Coldness is derived from `n_observations`,** never from a stored flag. Default
    values are never serialised as if they were measurements.
22. **Every mechanism publishes its effect into one health record** that a single loop
    prints each cycle. An acceptance criterion with no consumer in that record is not
    written. The old system had no evaluation stage at all — which is why four
    validators were dead and nobody noticed.

---

## The meta-rule

Eighteen of these twenty-two are **ownership or grounding** defects, not logic defects.
A rewrite fixes logic for free and reproduces ownership gaps exactly, because nobody
writes "and this is whose job" into a module.

So the first artifact of the rewrite is not code. It is the table naming, for every
adaptive quantity: its **one writer**, its readers, what **grounds** it, what is
**frozen** vs what moves, how it could fail **silently**, and the **canary** that makes
that failure loud.
