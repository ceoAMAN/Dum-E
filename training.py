from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import configs


# ── finite guards ───────────────────────────────────────────────────────────
# Each forces a host sync, so callers gate them behind a cadence (check_finite)
# rather than paying the stall every batch — a bad step is rare and the next
# checkpoint's validation catches drift.
def _is_finite_scalar(value: mx.array) -> bool:
    finite = mx.all(mx.isfinite(value))
    mx.eval(finite)
    return bool(finite.item())


def _tree_is_finite(tree) -> bool:
    flat = dict(tree_flatten(tree))
    if not flat:
        return True
    checks = [mx.all(mx.isfinite(v)) for v in flat.values()]
    all_finite = mx.all(mx.stack(checks))
    mx.eval(all_finite)
    return bool(all_finite.item())


def flatten_params(params) -> Optional[mx.array]:
    """Flatten a parameter pytree into one vector — used to compare expert weight
    matrices for peer repulsion. Returns None if empty."""
    flat = tree_flatten(params)
    if not flat:
        return None
    return mx.concatenate([v.reshape(-1) for _, v in flat])


# ── gate loss terms (all differentiable w.r.t. the gate via route_head) ──────
def compute_l_dom(domain_logits: mx.array, routing_density: mx.array) -> mx.array:
    """Cross-entropy: train the gate's 4-way domain head toward the true domain.

    The target must be an EXTERNAL label. Passing the gate's own domain output
    back in (topo.domain_proportions is derived from the same head) trains the
    head toward its own opinion, which locks in whatever the random
    initialisation happened to say. apply_gate_gradients therefore skips this
    term entirely when no label is supplied, rather than substituting a
    self-derived one."""
    log_probs = mx.log(mx.softmax(domain_logits) + 1e-10)
    target = routing_density / (mx.sum(routing_density) + 1e-8)
    return -mx.sum(target * log_probs)


def compute_l_eff_loss(route_logits: mx.array, active_ids: List[int], l_eff_targets: Dict[int, float]) -> mx.array:
    """Push the routing head's preference (over the experts that ran this batch)
    toward the efficiency distribution measured by Central. Cross-entropy between
    softmax(route_logits[active]) and the L1-normalised L_eff scores. Lagged by one
    batch by construction: the targets come from the batch that just finished."""
    if not active_ids:
        return mx.array(0.0, dtype=mx.float32)
    idx = mx.array(active_ids)
    log_pred = mx.log(mx.softmax(route_logits[idx]) + 1e-10)
    raw = mx.array([float(l_eff_targets.get(e, 0.0)) for e in active_ids], dtype=mx.float32)
    total = mx.sum(raw)
    # Uniform target if no efficiency signal yet (keeps the term finite, gradient ~0).
    target = mx.where(total > 1e-8, raw / (total + 1e-8), mx.ones_like(raw) / len(active_ids))
    return -mx.sum(target * log_pred)


def compute_l_rel(route_logits: mx.array, active_ids: List[int], staleness: Dict[int, float]) -> mx.array:
    """Penalise routing probability mass that lands on stale experts (high past
    R_i, low recent). staleness[eid] in [0,1]; minimising sum(prob * staleness)
    teaches the gate to stop coasting on experts that hold rank by inertia."""
    if not active_ids:
        return mx.array(0.0, dtype=mx.float32)
    idx = mx.array(active_ids)
    pred = mx.softmax(route_logits[idx])
    stale = mx.array([float(staleness.get(e, 0.0)) for e in active_ids], dtype=mx.float32)
    return mx.sum(pred * stale)


def _n_domains() -> int:
    """D = len(gating.DOMAINS), read live so adding a domain needs no edit here.
    Imported lazily to avoid a circular import with gating."""
    from gating import DOMAINS
    return max(1, len(DOMAINS))


def _gate_route_outputs(net, tokens: mx.array):
    """(domain_logits, route_logits) from the gate's pooled hidden state. Mirrors
    gating.GateModel.forward exactly so the gradient and the routing pass see the
    same transform — train/infer consistency."""
    backbone = net.backbone
    hidden = backbone.model(tokens.reshape(1, -1)) if hasattr(backbone, "model") else backbone(tokens.reshape(1, -1))
    mean_hidden = mx.clip(mx.mean(hidden[0], axis=0), -1e4, 1e4)
    mu = mx.mean(mean_hidden)
    sigma = mx.sqrt(mx.mean((mean_hidden - mu) ** 2) + 1e-8)
    domain_logits = ((mean_hidden - mu) / (sigma + 1e-8))[:_n_domains()]
    route_logits = net.route_head(mean_hidden)
    return domain_logits, route_logits


def apply_gate_gradients(
    gate_net,
    gate_optimizer,
    tokens: mx.array,
    lambdas: mx.array,
    routing_density: Optional[mx.array] = None,
    active_expert_ids: Optional[List[int]] = None,
    l_eff_targets: Optional[Dict[int, float]] = None,
    staleness: Optional[Dict[int, float]] = None,
    check_finite: bool = True,
) -> Dict[str, float]:
    """One L_gate step over the whole gate (backbone LoRA + route_head):

        L_gate = λ_eff·L_eff + λ_dom·L_dom + λ_rel·L_rel

    L_dom trains the domain head; L_eff/L_rel train the routing head. All three are
    real gradients on gate parameters (the routing head is what makes L_eff/L_rel
    differentiable — without it they were dead compute). MAML's lambdas weight them.
    """
    active = active_expert_ids or []
    targets = l_eff_targets or {}
    stale = staleness or {}
    terms: Dict[str, mx.array] = {}   # capture each L_gate term for benchmark logging

    def gate_loss_fn(net):
        domain_logits, route_logits = _gate_route_outputs(net, tokens)
        # No label -> no L_dom. L_eff and L_rel are grounded in r_i measured by
        # Central, so they are safe to run on the inference path; L_dom needs a
        # domain label the inference path does not have.
        l_dom = (compute_l_dom(domain_logits, routing_density)
                 if routing_density is not None else mx.array(0.0, dtype=mx.float32))
        l_eff = compute_l_eff_loss(route_logits, active, targets)
        l_rel = compute_l_rel(route_logits, active, stale)
        terms["l_eff"], terms["l_dom"], terms["l_rel"] = l_eff, l_dom, l_rel
        return lambdas[0] * l_eff + lambdas[1] * l_dom + lambdas[2] * l_rel

    loss, grads = nn.value_and_grad(gate_net, gate_loss_fn)(gate_net)
    # Finite guard is NON-optional before the update: a NaN/Inf gradient that slips
    # through corrupts the weights permanently (every later batch goes NaN). Skip
    # the step instead. check_finite only gates the cheaper post-update param sweep.
    if not _is_finite_scalar(loss) or not _tree_is_finite(grads):
        return {"total": 0.0, "l_eff": 0.0, "l_dom": 0.0, "l_rel": 0.0}
    grads, _ = optim.clip_grad_norm(grads, configs.GRAD_CLIP_NORM)
    gate_optimizer.update(gate_net, grads)
    mx.eval(gate_net.parameters(), gate_optimizer.state)
    if check_finite and not _tree_is_finite(gate_net.trainable_parameters()):
        return {"total": 0.0, "l_eff": 0.0, "l_dom": 0.0, "l_rel": 0.0}
    mx.eval(loss, terms["l_eff"], terms["l_dom"], terms["l_rel"])  # cheap scalars, already on the loss graph
    return {
        "total": float(loss.item()),
        "l_eff": float(terms["l_eff"].item()),
        "l_dom": float(terms["l_dom"].item()),
        "l_rel": float(terms["l_rel"].item()),
    }


def apply_expert_gradients(
    expert_model,
    expert_optimizer,
    tokens: mx.array,
    central_synthesis: mx.array,
    peer_weights: Optional[List[mx.array]] = None,
    l_div_weight: float = 0.0,
    web_target: Optional[mx.array] = None,
    web_weight: float = 0.0,
    check_finite: bool = True,
) -> Dict[str, float]:
    """One expert step toward Central's synthesis, plus EITHER peer repulsion or
    spiderweb attraction — never both, because they are opposite forces:

        outside the web:  L = MSE(hidden, synthesis) + λ_div · mean_j cos(W_i, W_j)
        inside the web:   L = MSE(hidden, synthesis) + w_i  · (1 − cos(W_i, target))

    The MSE pulls the expert's representation toward Central's synthesised view.

    REPULSION (bottom line, outside the web) pushes this expert away from its
    co-active peers so healthy experts keep covering different ground. It
    self-limits: as weights diverge the cosine → 0 and the gradient vanishes.

    ATTRACTION (inside the web) pulls a bottom-half / dormant expert toward
    `web_target`, the mean of the better half's weights. Also self-limiting, but
    from the other side: the expert climbs out of the bottom half, the caller
    stops passing a target, and repulsion resumes. Applying only one of the two
    per expert per batch is what stops them cancelling each other out.
    """
    peers = peer_weights or []
    terms: Dict[str, mx.array] = {}   # capture MSE and peer-similarity for benchmark logging

    def expert_loss_fn(model):
        token_batch = mx.array(tokens, dtype=mx.int32).reshape(1, -1)
        hidden_out = model.model(token_batch)
        if hidden_out.ndim == 3:
            hidden_mean = mx.mean(hidden_out[0], axis=0)
        elif hidden_out.ndim == 2:
            hidden_mean = mx.mean(hidden_out, axis=0)
        else:
            hidden_mean = hidden_out
        min_dim = min(hidden_mean.shape[0], central_synthesis.shape[0])
        mse = mx.mean((hidden_mean[:min_dim] - central_synthesis[:min_dim]) ** 2)
        terms["mse"] = mse
        loss = mse
        in_web = web_target is not None and web_weight > 0.0
        if in_web or (peers and l_div_weight > 0.0):
            w_i = flatten_params(model.trainable_parameters())
            if w_i is not None:
                w_i_n = w_i / (mx.linalg.norm(w_i) + 1e-8)
                if in_web:
                    # Attraction: squared distance to the better half's mean
                    # weights. Well-defined at W = 0 (gradient -> -2*target),
                    # unlike a cosine, so a dormant expert can actually be moved.
                    d = min(int(w_i.shape[0]), int(web_target.shape[0]))
                    terms["web_mse"] = mx.mean((w_i[:d] - web_target[:d]) ** 2)
                    # Reported only — stop_gradient keeps the singular d(cos)/dW
                    # at the origin out of the backward pass.
                    tn = mx.linalg.norm(web_target[:d])
                    terms["web_cos"] = mx.stop_gradient(
                        mx.sum(w_i_n[:d] * web_target[:d]) / (tn + 1e-8)
                    )
                    loss = loss + web_weight * terms["web_mse"]
                else:
                    sims = mx.stack([mx.sum(w_i_n * pj) for pj in peers])  # cos sim to each peer (pre-normalised, detached)
                    mean_sim = mx.mean(sims)                               # mean peer similarity (lower = more diverged)
                    terms["sim"] = mean_sim
                    loss = loss + l_div_weight * mean_sim
        return loss

    loss, grads = nn.value_and_grad(expert_model, expert_loss_fn)(expert_model)
    # Finite guard is non-optional before the update (see apply_gate_gradients).
    if not _is_finite_scalar(loss) or not _tree_is_finite(grads):
        return {"total": 0.0, "mse": 0.0, "l_div_sim": 0.0, "web_cos": 0.0, "web_mse": 0.0}
    grads, _ = optim.clip_grad_norm(grads, configs.GRAD_CLIP_NORM)
    expert_optimizer.update(expert_model, grads)
    mx.eval(expert_model.parameters(), expert_optimizer.state)
    if check_finite and not _tree_is_finite(expert_model.trainable_parameters()):
        return {"total": 0.0, "mse": 0.0, "l_div_sim": 0.0, "web_cos": 0.0, "web_mse": 0.0}
    sim_arr = terms.get("sim")
    web_arr = terms.get("web_cos")
    mse_arr = terms.get("web_mse")
    extras = [a for a in (sim_arr, web_arr, mse_arr) if a is not None]
    mx.eval(loss, terms["mse"], *extras)
    return {
        "total": float(loss.item()),
        "mse": float(terms["mse"].item()),
        "l_div_sim": float(sim_arr.item()) if sim_arr is not None else 0.0,
        # web_mse is the actual attraction term; web_cos is the readable monitor —
        # it climbs toward 1.0 as a dormant expert is pulled into the better
        # half's region, at which point the caller stops passing a target.
        "web_mse": float(mse_arr.item()) if mse_arr is not None else 0.0,
        "web_cos": float(web_arr.item()) if web_arr is not None else 0.0,
    }


def peer_weight_vector(model) -> Optional[mx.array]:
    """Normalised, detached flat weight vector of an expert, for use as a peer
    reference in another expert's repulsion term (no gradient flows back into it)."""
    w = flatten_params(model.trainable_parameters())
    if w is None:
        return None
    return mx.stop_gradient(w / (mx.linalg.norm(w) + 1e-8))


# ── spiderweb: rank-directed remediation for dormant experts ────────────────
# Every expert in a batch works its own fragment of the SAME input, so their work
# is directly comparable — that is what licenses peers to evaluate each other.
# The active set is split at the median; the better half becomes a reference the
# worse half is pulled toward. Because the fragments differ, this cannot mean
# copying an answer: pulling in WEIGHT space transfers how an expert processes,
# not what it concluded.
#
# The target population is the dormant pool — experts that are dead, never
# trained, or never routed to (a fresh LoRA has lora_b = 0, making it a
# mathematically exact no-op and a permanent bottom-half occupant). Context alone
# can't revive one: show it a good analysis and it answers better for exactly one
# batch, then reverts, because nothing was written. A gradient is what makes the
# improvement stick.
#
# Lifecycle: dormant -> routed in -> ranks bottom half -> pulled toward the top
# half (lora_b becomes non-zero, i.e. alive) -> climbs out -> web releases ->
# L_div takes over and pushes it to its own niche. Attraction and repulsion never
# apply to the same expert on the same batch, so they cannot cancel.
def nl(n: int) -> int:
    """NL — the LOWER of the two naturals bracketing sqrt(n).

    sqrt(n) is irrational for non-square n, so it can never be a count directly;
    it falls strictly between NL = floor(sqrt(n)) and NU = ceil(sqrt(n)). The
    selection rule `NL < sqrt(k) < NU -> take NL` resolves that to the lower one.
    n=10 -> sqrt=3.162 -> NL=3, NU=4, choose 3. For perfect squares NL == NU."""
    return max(1, math.floor(math.sqrt(max(1, int(n)))))


def nu(n: int) -> int:
    """NU = ceil(sqrt(n)) — the upper bracket. n=10 -> 4. NL is what gets chosen;
    NU is the bound the rule tests against."""
    return max(1, math.ceil(math.sqrt(max(1, int(n)))))


def tier_split(ranked_expert_ids: List[int]) -> tuple:
    """Three tiers over a ranked (best-first) set, each NL = floor(sqrt(n)) wide:

        best   [0:NL]      — apply pressure, receive none
        middle [NL:2NL]    — strong in-domain but underused; left alone, and the
                             recruiting pool for new/!general domains
        rest   [2NL:]      — receive pressure from best

    Slicing degrades safely: when 2*NL >= n the rest tier is simply empty."""
    ids = list(ranked_expert_ids)
    k = len(ids)
    if k < 2:
        return (ids, [], [])
    w = nl(k)
    return (ids[:w], ids[w:2 * w], ids[2 * w:])


def split_active_set(ranked_expert_ids: List[int]) -> tuple:
    """Median split of the batch's active experts, best-first. An odd count sends
    the middle expert to the BEST half (k=5 -> 3 best / 2 worst), so the reference
    side is never the smaller one. k=1 yields an empty worst half — nothing to
    pull from, so no web forms."""
    ids = list(ranked_expert_ids)
    k = len(ids)
    if k < 2:
        return (ids, [])
    n_best = (k + 1) // 2          # ceil(k/2)
    return (ids[:n_best], ids[n_best:])


def expert_weight_vector(model) -> Optional[mx.array]:
    """RAW (unnormalised), detached flat weight vector — the form spiderweb needs.
    Distinct from peer_weight_vector, which normalises because repulsion is a
    cosine; attraction must keep magnitude so a dead expert also learns the SCALE
    of a working one, not just its direction."""
    w = flatten_params(model.trainable_parameters())
    if w is None:
        return None
    return mx.stop_gradient(w)


def spiderweb_target(best_weight_vectors: List[mx.array]) -> Optional[mx.array]:
    """The reference a bottom-half expert is pulled toward: the mean of the top
    half's RAW weight vectors (from expert_weight_vector), detached. Averaging the
    whole better half, rather than copying the single best, keeps the target from
    being one expert's idiosyncrasies.

    Deliberately NOT normalised. Attraction is squared distance, not cosine,
    because cosine is singular exactly where this mechanism is aimed: a dormant
    expert has lora_b = 0, so ‖W‖ = 0, and d/dW of cos(W, t) divides by ‖W‖ —
    the gradient goes non-finite, the finite guard rejects the step, and the dead
    expert stays dead forever. Squared distance has gradient 2(W − t), which at
    W = 0 is simply −2t: a clean pull toward the better half."""
    vecs = [v for v in best_weight_vectors if v is not None]
    if not vecs:
        return None
    dim = min(int(v.shape[0]) for v in vecs)
    stacked = mx.stack([v[:dim] for v in vecs], axis=0)
    return mx.stop_gradient(mx.mean(stacked, axis=0))


def spiderweb_pressure(rank_index: int, n_worst: int, is_dormant: bool = False) -> float:
    """Pressure for the rank_index-th expert of the worst half (0 = least bad).
    Scales to 1.0 at the very bottom; a dormant/never-trained expert is pinned at
    full pressure regardless of rank, since it has no niche to protect."""
    if is_dormant:
        return 1.0
    if n_worst <= 1:
        return 1.0
    return float(rank_index + 1) / float(n_worst)


# ── the grounding objective: real next-token cross-entropy ───────────────────
# This is the fix for the hollow core. Every OTHER signal in the system (r_i,
# L_eff, expert MSE) is measured against Central's OWN hidden state, so nothing
# was ever checked against reality — Central itself never received a gradient in
# the entire 994k-token run. compute_central_ce is the reality check: teacher-
# forced CE of Central over the REAL continuation tokens of real corpus text.
# It's the only loss in the codebase anchored to something outside the model.
def compute_central_ce(model, input_ids: mx.array, n_context: int,
                       per_token: bool = False) -> mx.array:
    """Teacher-forced cross-entropy over the TARGET region only.

    input_ids = [ prompt+expert context (n_context tokens) | real target tokens ]
    (built by central.build_training_ids). Next-token prediction: logits[t]
    predicts input_ids[t+1], so the predictions for the target tokens at
    positions [n_context, T) come from logits[n_context-1, T-1). We score ONLY
    those — the model is never rewarded for parroting the prompt or the expert
    context, only for predicting the real continuation. Returned on-graph (no
    mx.eval) so value_and_grad can backprop it; measurement callers eval it.

    `per_token=True` returns the un-reduced vector, one CE per target token,
    instead of the mean. The mean is what the gradient wants, but it is blind to
    exactly the failure we care about: a handful of badly wrong tokens disappear
    into an average over hundreds of fine ones, so a confident fabrication in the
    middle of an otherwise correct answer is invisible. Attributing hallucination
    to token types needs the un-reduced vector, and it comes out of the same
    forward pass at no extra cost."""
    logits = model(input_ids.reshape(1, -1))
    logits = logits[0] if logits.ndim == 3 else logits          # (T, V)
    t_total = int(logits.shape[0])
    start = max(1, int(n_context))
    if start >= t_total:                                        # no target region
        return mx.zeros((0,), dtype=mx.float32) if per_token else mx.array(0.0, dtype=mx.float32)
    pred = logits[start - 1:t_total - 1, :]                     # predictions for targets
    tgt = input_ids[start:t_total]                             # the real target tokens
    return nn.losses.cross_entropy(pred, tgt, reduction="none" if per_token else "mean")


def apply_central_pretrain(central, central_optimizer, input_text: str,
                           target_ids: List[int], check_finite: bool = True) -> Dict[str, float]:
    """One Central training step — PLAIN next-token CE on real text, no expert
    context.

    Central is trained FIRST, on its own, over its own token budget; only after
    that does the joint phase run, where it synthesises and is no longer taking
    gradients. So the training signal here is ordinary language modelling: given
    the question, predict the true continuation.

    (This replaces a per-batch dual objective that trained Central as model AND
    synthesiser simultaneously. Wrong shape for this design — and freezing
    Central through the joint phase is the stronger arrangement anyway, because
    r_i is measured against Central's loss: if Central keeps moving, the yardstick
    moves with it and expert scores from different batches stop being
    comparable.)"""
    central.load()
    model = central.model
    input_ids, n_context = central.build_training_ids(input_text, [], target_ids)
    if len(input_ids) <= n_context:
        return {"total": 0.0, "ce": 0.0}
    ids = mx.array(input_ids)

    def loss_fn(m):
        return compute_central_ce(m, ids, n_context)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    if not _is_finite_scalar(loss) or not _tree_is_finite(grads):
        return {"total": 0.0, "ce": 0.0}
    grads, _ = optim.clip_grad_norm(grads, configs.GRAD_CLIP_NORM)
    central_optimizer.update(model, grads)
    mx.eval(model.parameters(), central_optimizer.state)
    if check_finite and not _tree_is_finite(model.trainable_parameters()):
        return {"total": 0.0, "ce": 0.0}
    return {"total": float(loss.item()), "ce": float(loss.item())}


def grounded_r_i(central, input_text: str, expert_outputs: List[Dict], target_ids: List[int],
                 sample: Optional[List[int]] = None) -> Dict[int, float]:
    """Per-expert contribution, measured against REAL text via leave-one-out.

        loss_with    = CE over the true continuation, all experts in context
        loss_without = CE with THIS expert's analysis removed
        r_i          = sigmoid(loss_without - loss_with)

    Positive delta means removing the expert made the real continuation harder to
    predict — it was carrying information. This is the fix for the circular
    reward: `central.compute_r_i` scores cosine against Central's OWN hidden
    state, so an expert that confidently agrees with a hallucinated synthesis
    earns a high score. Here the reference is the actual next tokens of real
    text, which no amount of agreement can fake.

    Cost is len(sample)+1 forwards, so pass `sample` to amortise — scoring one
    expert per batch is enough to anchor routing, since the signal only needs to
    be periodic to keep the head off the throughput heuristic it currently
    learns. Only computable where a true continuation exists (training/eval); a
    live user question has no known answer, so inference still falls back to the
    cosine heuristic."""
    if not expert_outputs or not target_ids:
        return {}
    with_all = central_ce_value(central, input_text, expert_outputs, target_ids)
    if with_all is None:
        return {}
    ids = [int(e.get("expert_id", i)) for i, e in enumerate(expert_outputs)]
    chosen = set(sample) if sample else set(ids)
    out: Dict[int, float] = {}
    for eid, eo in zip(ids, expert_outputs):
        if eid not in chosen:
            continue
        without = [x for x in expert_outputs if x is not eo]
        v = central_ce_value(central, input_text, without, target_ids)
        if v is None:
            continue
        out[eid] = central.compute_grounded_r_i(v, with_all)
    return out


def probe_central_capacity(central, samples: List[tuple], start_tokens: Optional[int] = None,
                           step: float = 2.0, max_probes: int = 24) -> Dict[str, float]:
    """Central's OWN apex-nadir, over input size — and the minimum cap it yields.

    Central has an overfit ceiling and an underfit floor exactly as the experts
    do, and nothing measured them. Procedure:

      1. start at a token count (random within the corpus range if unspecified)
      2. INCREASE until it overfits — loss stops improving as context grows
      3. DECREASE until it underfits — loss climbs sharply as context is starved
      4. regress over EVERY count touched on the way (not just the endpoints)
      5. the minimum cap is where the fitted loss curve stops improving on the
         low side: below it Central lacks context to answer alone

    `samples` are (input_text, target_ids) pairs. Returns the fitted band plus
    `min_cap`, which is what the A/B decision should key on: an input below the
    cap has too little to spread across experts, so Central handles it solo;
    above it, Timeline B. That replaces deciding A/B from gate confidence, which
    is entropy over a z-scored slice of hidden state and tracks nothing about
    whether an input actually needs experts."""
    if not samples:
        return {}
    central.load()
    limit = central.context_limit()
    lo_bound = 8
    hi_bound = max(lo_bound + 1, min(limit, max(len(t) for _, t in samples) * 8 or limit))
    t = int(start_tokens or max(lo_bound, min(hi_bound, 64)))
    seen: Dict[int, float] = {}

    def loss_at(n: int) -> Optional[float]:
        vals = []
        for text, tgt in samples:
            ids = central.tokenizer.encode(text)[:n]
            if not ids or not tgt:
                continue
            v = central_ce_value(central, central.tokenizer.decode(ids), [], tgt)
            if v is not None and math.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    # 2. climb until loss stops improving => overfit / saturation
    best = loss_at(t)
    if best is None:
        return {}
    seen[t] = best
    cur = t
    for _ in range(max_probes // 2):
        nxt = int(min(hi_bound, cur * step))
        if nxt <= cur:
            break
        v = loss_at(nxt)
        if v is None:
            break
        seen[nxt] = v
        cur = nxt
        if v >= best:                    # no longer improving -> apex reached
            break
        best = v
    # 3. descend until loss degrades sharply => underfit
    cur = t
    for _ in range(max_probes // 2):
        nxt = int(max(lo_bound, cur / step))
        if nxt >= cur:
            break
        v = loss_at(nxt)
        if v is None:
            break
        seen[nxt] = v
        cur = nxt
    if len(seen) < 3:
        return {}
    # 4. regress over every count touched
    xs = np.log(np.array(sorted(seen), dtype=np.float64))
    ys = np.array([seen[int(round(math.exp(x)))] for x in xs], dtype=np.float64)
    try:
        coeffs = np.polyfit(xs, ys, 2)
    except Exception:
        return {}
    grid = np.linspace(xs.min(), xs.max(), 128)
    fit = np.polyval(coeffs, grid)
    apex = float(math.exp(grid[int(np.argmin(fit))]))     # best loss = the operating peak
    # 5. min cap: smallest size still within one std of the best fitted loss
    tol = float(np.std(ys)) if len(ys) > 1 else 0.0
    ok = np.where(fit <= fit.min() + tol)[0]
    min_cap = float(math.exp(grid[int(ok[0])])) if len(ok) else apex
    return {
        "min_cap": min_cap,
        "apex": apex,
        "nadir": float(math.exp(xs.min())),
        "ceiling": float(math.exp(xs.max())),
        "probes": float(len(seen)),
    }


def central_ce_value(central, input_text: str, expert_outputs: List[Dict], target_ids: List[int]) -> Optional[float]:
    """Scalar CE of Central over the real targets for a GIVEN context — no gradient.
    Run it twice (with vs without one expert's output_text in expert_outputs) and
    feed the two losses to central.compute_grounded_r_i to get that expert's REAL
    contribution: did it actually make the true continuation more predictable?
    Returns None when there's no target region to score.

    Cost note: a full leave-one-out over k experts is k+1 forwards per batch — real
    money on a 7B. Amortise it (score a sampled expert per batch, or every N
    batches) rather than every expert every batch; the grounded signal only needs
    to be periodic to anchor the routing head off the throughput heuristic."""
    central.load()
    input_ids, n_context = central.build_training_ids(input_text, expert_outputs, target_ids)
    if len(input_ids) <= n_context:
        return None
    loss = compute_central_ce(central.model, mx.array(input_ids), n_context)
    mx.eval(loss)
    return float(loss.item())
