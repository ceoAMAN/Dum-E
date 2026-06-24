from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
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
def compute_l_dom(gate_domain_logits: mx.array, routing_memory_density: mx.array) -> mx.array:
    log_probs = mx.log(mx.softmax(gate_domain_logits) + 1e-10)
    target = routing_memory_density / (mx.sum(routing_memory_density) + 1e-8)
    return -mx.sum(target * log_probs)
def apply_gate_gradients(
    gate_model,
    gate_optimizer,
    tokens: mx.array,
    lambdas: mx.array,
    routing_density: mx.array,
):
    # The gate only learns from l_dom (domain-routing cross-entropy). The former
    # l_eff/l_rel/l_div terms were computed from values DETACHED from the gate's
    # compute graph (precomputed expert scores / expert hidden states), so their
    # gradient w.r.t. the gate was provably zero — they only burned compute. This
    # is gradient-identical to the old 4-term loss, just without the dead work.
    # lambdas[1] (l_dom's MAML weight) is kept so MAML still modulates routing LR.
    def gate_loss_fn(model):
        if hasattr(model, 'model'):
            hidden = model.model(tokens.reshape(1, -1))
        else:
            hidden = model(tokens.reshape(1, -1))
        mean_hidden = mx.mean(hidden[0], axis=0)
        mean_hidden = mx.clip(mean_hidden, -1e4, 1e4)
        # Match normalisation from gating.py forward() — keep train/infer consistent.
        mu = mx.mean(mean_hidden)
        sigma = mx.sqrt(mx.mean((mean_hidden - mu) ** 2) + 1e-8)
        normed = (mean_hidden - mu) / (sigma + 1e-8)
        domain_logits = normed[:4]   # 4 domain slots
        return lambdas[1] * compute_l_dom(domain_logits, routing_density)
    loss_and_grad_fn = nn.value_and_grad(gate_model, gate_loss_fn)
    loss, grads = loss_and_grad_fn(gate_model)
    if not _is_finite_scalar(loss) or not _tree_is_finite(grads):
        return 0.0
    gate_optimizer.update(gate_model, grads)
    mx.eval(gate_model.parameters(), gate_optimizer.state)
    if not _tree_is_finite(gate_model.parameters()):
        return 0.0
    return float(loss.item())
def apply_expert_gradients(
    expert_model,
    expert_optimizer,
    tokens: mx.array,
    central_synthesis: mx.array
):
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
        return mx.mean((hidden_mean[:min_dim] - central_synthesis[:min_dim]) ** 2)
    loss_and_grad_fn = nn.value_and_grad(expert_model, expert_loss_fn)
    loss, grads = loss_and_grad_fn(expert_model)
    if not _is_finite_scalar(loss) or not _tree_is_finite(grads):
        return 0.0
    expert_optimizer.update(expert_model, grads)
    mx.eval(expert_model.parameters(), expert_optimizer.state)
    if not _tree_is_finite(expert_model.trainable_parameters()):
        return 0.0
    return float(loss.item())
