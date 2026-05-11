from __future__ import annotations
from typing import List
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import configs
from vectors import dot_product_similarity_matrix, exponential_decay_weights
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
def compute_l_eff_loss(l_eff_scores: mx.array, selected_mask: mx.array) -> mx.array:
    selected_scores = l_eff_scores * selected_mask
    n_selected = mx.sum(selected_mask) + 1e-8
    mean_score = mx.sum(mx.abs(selected_scores)) / n_selected
    return -mx.log(mx.maximum(mean_score, mx.array(configs.L_EFF_EPS)) + configs.L_EFF_EPS)
def compute_l_dom(gate_domain_logits: mx.array, routing_memory_density: mx.array) -> mx.array:
    log_probs = mx.log(mx.softmax(gate_domain_logits) + 1e-10)
    target = routing_memory_density / (mx.sum(routing_memory_density) + 1e-8)
    return -mx.sum(target * log_probs)
def compute_l_rel(r_i_history: List[mx.array], gamma: float = configs.L_REL_GAMMA) -> mx.array:
    if len(r_i_history) < 2:
        return mx.array(0.0)
    n = len(r_i_history)
    weights = exponential_decay_weights(n, gamma)
    weighted_sum = mx.array(0.0)
    for i, r_i in enumerate(r_i_history):
        if isinstance(r_i, (int, float)):
            r_i = mx.array(float(r_i))
        weighted_sum = weighted_sum + weights[i] * r_i
    return weighted_sum / n
def compute_l_div(weight_snapshots: List[mx.array]) -> mx.array:
    if len(weight_snapshots) < 2:
        return mx.array(0.0)
    current = weight_snapshots[-1]
    total_sim = mx.array(0.0)
    count = 0
    for prev in weight_snapshots[:-1]:
        min_dim = min(current.shape[0], prev.shape[0])
        c = current[:min_dim].reshape(1, -1)
        p = prev[:min_dim].reshape(1, -1)
        c_norm = c / (mx.linalg.norm(c) + 1e-8)
        p_norm = p / (mx.linalg.norm(p) + 1e-8)
        sim = mx.matmul(c_norm, p_norm.T)
        total_sim = total_sim + mx.mean(sim)
        count += 1
    if count == 0:
        return mx.array(0.0)
    return total_sim / count
def compute_l_gate(
    lambdas: mx.array,
    l_eff: mx.array,
    l_dom: mx.array,
    l_rel: mx.array,
    l_div: mx.array,
) -> mx.array:
    return lambdas[0] * l_eff + lambdas[1] * l_dom + lambdas[2] * l_rel + lambdas[3] * l_div
def compute_dot_product_peer_gradients(expert_weights: List[mx.array]) -> mx.array:
    sim_matrix = dot_product_similarity_matrix(expert_weights)
    n = len(expert_weights)
    identity = mx.eye(n)
    off_diagonal = sim_matrix * (1.0 - identity)
    repulsion_loss = mx.sum(off_diagonal ** 2)
    return repulsion_loss
def apply_gate_gradients(
    gate_model,
    gate_optimizer,
    tokens: mx.array,
    lambdas: mx.array,
    l_eff_scores: mx.array,
    selected_mask: mx.array,
    routing_density: mx.array,
    r_i_history: List[mx.array],
    weight_snapshots: List[mx.array]
):
    def gate_loss_fn(model):
        hidden = model(tokens.reshape(1, -1))
        mean_hidden = mx.mean(hidden[0], axis=0)
        mean_hidden = mx.clip(mean_hidden, -1e4, 1e4)
        domain_logits = mean_hidden[:configs.K_MAX]
        l_eff = compute_l_eff_loss(l_eff_scores, selected_mask)
        l_dom = compute_l_dom(domain_logits, routing_density)
        l_rel = compute_l_rel(r_i_history)
        l_div = compute_l_div(weight_snapshots)
        return compute_l_gate(lambdas, l_eff, l_dom, l_rel, l_div)
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
