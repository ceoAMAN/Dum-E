from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import configs
@dataclass
class KVelocityRecord:
    domain: str
    k_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    window_size: int = 1000
@dataclass
class MAMLState:
    lambdas: mx.array = field(default_factory=lambda: mx.array(configs.LAMBDA_INIT, dtype=mx.float32))
    token_count: int = 0
    last_outer_token: int = 0
    second_order_enabled: bool = False
    outer_step_count: int = 0
    k_velocity_records: Dict[str, KVelocityRecord] = field(default_factory=dict)
    instability_streak: int = 0
class MAMLOptimiser:
    def __init__(self, gate_model: nn.Module):
        self.gate_model = gate_model
        self.lambdas = mx.array(configs.LAMBDA_INIT, dtype=mx.float32)
        self.alpha_lr = configs.ALPHA_LR
        self.beta_lr = configs.BETA_LR
        # Dedicated rate for the loss-weight (lambda) meta-update. beta_lr stays
        # tied to alpha_lr (10:1) for the gate-parameter MAML; the lambdas need a
        # much larger rate or they never move off the uniform init.
        self.lambda_meta_lr = getattr(configs, "LAMBDA_META_LR", configs.BETA_LR)
        self._validate_lr_ratio()
        self.state = MAMLState(lambdas=self.lambdas)
        self._prev_lambda_norm: Optional[float] = None
    def _normalise_lambdas(self, lambdas: mx.array) -> mx.array:
        # Project onto the simplex with a per-element floor so no loss term is
        # ever fully zeroed: normalised = floor + (1 - n*floor) * base.
        n = int(lambdas.shape[0])
        floor = float(getattr(configs, "LAMBDA_FLOOR", 0.0))
        floor = max(0.0, min(floor, 1.0 / max(n, 1)))  # guard: n*floor <= 1
        clipped = mx.maximum(lambdas, mx.array(0.0, dtype=mx.float32))
        total = mx.sum(clipped) + 1e-8
        base = clipped / total
        normalised = floor + (1.0 - n * floor) * base
        mx.eval(normalised)
        return normalised
    def _validate_lr_ratio(self):
        ratio = self.beta_lr / self.alpha_lr
        assert abs(ratio - 0.1) < 1e-9, (
            f"beta_lr must equal alpha_lr * 0.1. Got ratio={ratio:.6f}."
        )
    def inner_step(self, gate_params: Dict, lambdas: mx.array, compute_l_gate_fn) -> Dict:
        def loss_fn(params):
            return compute_l_gate_fn(params, lambdas)
        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(gate_params)
        theta_prime = {k: v - self.alpha_lr * grads[k] for k, v in gate_params.items()}
        mx.eval(theta_prime)
        return theta_prime
    def outer_step(self, theta_prime: Dict, compute_l_meta_fn) -> mx.array:
        if self.state.second_order_enabled:
            updated = self._outer_step_second_order(theta_prime, compute_l_meta_fn)
        else:
            updated = self._outer_step_fomaml(theta_prime, compute_l_meta_fn)
        self._check_instability(updated)
        updated = self._normalise_lambdas(updated)
        self.lambdas = updated
        self.state.lambdas = updated
        self.state.outer_step_count += 1
        mx.eval(self.lambdas)
        return self.lambdas
    def _outer_step_fomaml(self, theta_prime: Dict, compute_l_meta_fn) -> mx.array:
        def meta_loss(lam):
            return compute_l_meta_fn(theta_prime, lam)
        lambda_grad_fn = mx.grad(meta_loss)
        lambda_grads = lambda_grad_fn(self.lambdas)
        return self.lambdas - self.lambda_meta_lr * lambda_grads
    def _outer_step_second_order(self, theta_prime: Dict, compute_l_meta_fn) -> mx.array:
        def meta_loss(lam):
            return compute_l_meta_fn(theta_prime, lam)
        meta_value = meta_loss(self.lambdas)
        _, lambda_vjps = mx.vjp(
            meta_loss,
            [self.lambdas],
            [mx.ones_like(meta_value)],
        )
        lambda_grads = lambda_vjps[0]
        return self.lambdas - self.lambda_meta_lr * lambda_grads
    def _check_instability(self, new_lambdas: mx.array):
        new_norm = float(mx.sum(mx.abs(new_lambdas)).item())
        if self._prev_lambda_norm is not None:
            delta = abs(new_norm - self._prev_lambda_norm)
            if delta > 0.5 * self._prev_lambda_norm:
                self.state.instability_streak += 1
            else:
                self.state.instability_streak = 0
            if self.state.instability_streak >= 3:
                self.lambdas = mx.array(configs.LAMBDA_INIT, dtype=mx.float32)
                self.state.instability_streak = 0
        self._prev_lambda_norm = new_norm
    def upgrade_to_second_order(self):
        self.state.second_order_enabled = True
    def should_run_outer_loop(self, current_token_count: int, last_outer_token: int) -> bool:
        return (current_token_count - last_outer_token) >= configs.OUTER_LOOP_TOKEN_INTERVAL
    def run_outer_step_from_metrics(
        self,
        domain: str,
        k_value: int,
        reconstruction_entropy: float,
        timeline_a_rate: float,
        cluster_count: int,
    ) -> mx.array:
        recent_velocity = self.compute_k_velocity(domain)
        if not np.isfinite(reconstruction_entropy):
            reconstruction_entropy = 0.0
        if not np.isfinite(timeline_a_rate):
            timeline_a_rate = 0.0
        k_signal = float(max(k_value, 0)) / max(configs.K_MAX, 1)
        entropy_signal = float(np.tanh(max(reconstruction_entropy, 0.0) / 10.0))
        fast_path_gap = float(max(0.0, 1.0 - timeline_a_rate))
        cluster_gap = 1.0 / max(cluster_count + 1, 1)
        if recent_velocity is not None:
            velocity_penalty = max(0.0, recent_velocity * configs.OUTER_LOOP_TOKEN_INTERVAL)
        else:
            velocity_penalty = 0.0
        targets = mx.array(
            [
                k_signal + velocity_penalty,
                entropy_signal,
                fast_path_gap,
                cluster_gap,
            ],
            dtype=mx.float32,
        )
        def compute_l_meta_fn(_theta_prime, lam):
            return mx.sum(lam * targets)
        updated = self.outer_step({}, compute_l_meta_fn)
        if self.should_upgrade_to_second_order():
            self.upgrade_to_second_order()
        return updated
    def record_k(self, domain: str, k_value: int, token_count: int):
        if domain not in self.state.k_velocity_records:
            self.state.k_velocity_records[domain] = KVelocityRecord(domain=domain)
        self.state.k_velocity_records[domain].k_history.append((token_count, k_value))
    def compute_k_velocity(self, domain: str, window: int = 1000) -> Optional[float]:
        if domain not in self.state.k_velocity_records:
            return None
        history = list(self.state.k_velocity_records[domain].k_history)
        if len(history) < 2:
            return None
        recent_cutoff = history[-1][0] - window
        recent = [k for t, k in history if t >= recent_cutoff]
        older = [k for t, k in history if t < recent_cutoff]
        if not recent or not older:
            return None
        return float((np.mean(recent) - np.mean(older)) / window)
    def k_velocity_is_sufficient(self, domain: str, threshold: float = -0.00010) -> bool:
        velocity = self.compute_k_velocity(domain)
        if velocity is None:
            return True
        return velocity < threshold
    def benchmark_k_velocity_all_domains(self, required_decrease: float = 0.10) -> Dict[str, bool]:
        results = {}
        for domain in self.state.k_velocity_records:
            velocity = self.compute_k_velocity(domain)
            if velocity is None:
                results[domain] = True
                continue
            results[domain] = velocity < 0 and abs(velocity) * 1000 >= required_decrease
        return results
    def should_upgrade_to_second_order(self) -> bool:
        if self.state.second_order_enabled:
            return False
        benchmark = self.benchmark_k_velocity_all_domains()
        insufficient = [d for d, passing in benchmark.items() if not passing]
        return len(insufficient) > 0 and self.state.outer_step_count >= 10
    def log_k_velocity_all_domains(self) -> Dict[str, Optional[float]]:
        results = {}
        for domain in self.state.k_velocity_records:
            results[domain] = self.compute_k_velocity(domain)
        return results
    def save(self):
        mx.savez(
            configs.LAMBDA_SAVE_PATH,
            lambdas=self.lambdas,
            outer_step_count=mx.array(self.state.outer_step_count),
            second_order_enabled=mx.array(int(self.state.second_order_enabled)),
        )
    def load(self):
        try:
            data = mx.load(configs.LAMBDA_SAVE_PATH)
            self.lambdas = data["lambdas"]
            self.state.lambdas = self.lambdas
            self.state.outer_step_count = int(data["outer_step_count"].item())
            self.state.second_order_enabled = bool(int(data["second_order_enabled"].item()))
            mx.eval(self.lambdas)
        except Exception:
            pass
    def get_lambdas(self) -> mx.array:
        return self.lambdas
