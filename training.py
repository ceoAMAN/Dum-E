from __future__ import annotations
from typing import Dict, List, Optional
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


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
    """Cross-entropy: train the gate's 4-way domain head toward the true domain."""
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


def _gate_route_outputs(net, tokens: mx.array):
    """(domain_logits, route_logits) from the gate's pooled hidden state. Mirrors
    gating.GateModel.forward exactly so the gradient and the routing pass see the
    same transform — train/infer consistency."""
    backbone = net.backbone
    hidden = backbone.model(tokens.reshape(1, -1)) if hasattr(backbone, "model") else backbone(tokens.reshape(1, -1))
    mean_hidden = mx.clip(mx.mean(hidden[0], axis=0), -1e4, 1e4)
    mu = mx.mean(mean_hidden)
    sigma = mx.sqrt(mx.mean((mean_hidden - mu) ** 2) + 1e-8)
    domain_logits = ((mean_hidden - mu) / (sigma + 1e-8))[:4]
    route_logits = net.route_head(mean_hidden)
    return domain_logits, route_logits


def apply_gate_gradients(
    gate_net,
    gate_optimizer,
    tokens: mx.array,
    lambdas: mx.array,
    routing_density: mx.array,
    active_expert_ids: Optional[List[int]] = None,
    l_eff_targets: Optional[Dict[int, float]] = None,
    staleness: Optional[Dict[int, float]] = None,
    check_finite: bool = True,
) -> float:
    """One L_gate step over the whole gate (backbone LoRA + route_head):

        L_gate = λ_eff·L_eff + λ_dom·L_dom + λ_rel·L_rel

    L_dom trains the domain head; L_eff/L_rel train the routing head. All three are
    real gradients on gate parameters (the routing head is what makes L_eff/L_rel
    differentiable — without it they were dead compute). MAML's lambdas weight them.
    """
    active = active_expert_ids or []
    targets = l_eff_targets or {}
    stale = staleness or {}

    def gate_loss_fn(net):
        domain_logits, route_logits = _gate_route_outputs(net, tokens)
        l_dom = compute_l_dom(domain_logits, routing_density)
        l_eff = compute_l_eff_loss(route_logits, active, targets)
        l_rel = compute_l_rel(route_logits, active, stale)
        return lambdas[0] * l_eff + lambdas[1] * l_dom + lambdas[2] * l_rel

    loss, grads = nn.value_and_grad(gate_net, gate_loss_fn)(gate_net)
    if check_finite and (not _is_finite_scalar(loss) or not _tree_is_finite(grads)):
        return 0.0
    gate_optimizer.update(gate_net, grads)
    mx.eval(gate_net.parameters(), gate_optimizer.state)
    if check_finite and not _tree_is_finite(gate_net.trainable_parameters()):
        return 0.0
    return float(loss.item())


def apply_expert_gradients(
    expert_model,
    expert_optimizer,
    tokens: mx.array,
    central_synthesis: mx.array,
    peer_weights: Optional[List[mx.array]] = None,
    l_div_weight: float = 0.0,
    check_finite: bool = True,
) -> float:
    """One expert step toward Central's synthesis, plus optional peer repulsion:

        L_expert = MSE(expert_hidden, synthesis) + λ_div · mean_j cos(W_i, W_j)

    The MSE pulls the expert's representation toward Central's synthesised view;
    the repulsion term pushes this expert's weights away from its co-active peers
    (peer_weights are detached vectors of the OTHER active experts). It self-limits:
    as weights become dissimilar the cosine → 0 and its gradient vanishes — experts
    specialise instead of collapsing into clones.
    """
    peers = peer_weights or []

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
        loss = mx.mean((hidden_mean[:min_dim] - central_synthesis[:min_dim]) ** 2)
        if peers and l_div_weight > 0.0:
            w_i = flatten_params(model.trainable_parameters())
            if w_i is not None:
                w_i_n = w_i / (mx.linalg.norm(w_i) + 1e-8)
                sims = [mx.sum(w_i_n * pj) for pj in peers]   # pj pre-normalised + detached
                loss = loss + l_div_weight * (mx.add(*sims) / len(sims) if len(sims) > 1 else sims[0])
        return loss

    loss, grads = nn.value_and_grad(expert_model, expert_loss_fn)(expert_model)
    if check_finite and (not _is_finite_scalar(loss) or not _tree_is_finite(grads)):
        return 0.0
    expert_optimizer.update(expert_model, grads)
    mx.eval(expert_model.parameters(), expert_optimizer.state)
    if check_finite and not _tree_is_finite(expert_model.trainable_parameters()):
        return 0.0
    return float(loss.item())


def peer_weight_vector(model) -> Optional[mx.array]:
    """Normalised, detached flat weight vector of an expert, for use as a peer
    reference in another expert's repulsion term (no gradient flows back into it)."""
    w = flatten_params(model.trainable_parameters())
    if w is None:
        return None
    return mx.stop_gradient(w / (mx.linalg.norm(w) + 1e-8))
