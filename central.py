from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, List
import mlx.core as mx
import numpy as np
import configs
from apex_nadir_convolution import ApexNadirConvolution
@dataclass
class CentralOutput:
    synthesis_text: str
    synthesis_hidden: mx.array
    contribution_hidden: mx.array
    expert_scores: Dict[int, float]
    expert_tkl: Dict[int, float]
    reconstruction_entropy: float
    send_to_user: bool
class CentralModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
    def load(self):
        if self._loaded:
            return
        from mlx_lm import load as mlx_load
        self.model, self.tokenizer = mlx_load(configs.CENTRAL_MODEL_ID)
        self._loaded = True
    def _compute_hidden_mean(self, token_ids: List[int]) -> mx.array:
        tokens = mx.array([token_ids[:configs.MAX_SEQ_LEN]])
        if hasattr(self.model, "model"):
            hidden_out = self.model.model(tokens)
            mx.eval(hidden_out)
        else:
            hidden_out = self.model(tokens)
            mx.eval(hidden_out)
        if hidden_out.ndim == 3:
            hidden_mean = mx.mean(hidden_out[0], axis=0)
        else:
            hidden_mean = mx.mean(hidden_out, axis=0)
        mx.eval(hidden_mean)
        return hidden_mean
    def _build_input_ids(self, original_input: str, expert_outputs: List[Dict[str, Any]]) -> List[int]:
        limit = configs.MAX_SEQ_LEN
        input_ids = self.tokenizer.encode(original_input)[:limit]
        if len(input_ids) >= limit:
            return input_ids
        newline_ids = self.tokenizer.encode("\n")
        for eo in expert_outputs:
            text = str(eo.get("output_text", ""))
            if not text:
                continue
            part_ids = self.tokenizer.encode(text)
            remaining = limit - len(input_ids)
            if remaining <= 0:
                break
            if newline_ids:
                input_ids.extend(newline_ids[:remaining])
                remaining = limit - len(input_ids)
                if remaining <= 0:
                    break
            input_ids.extend(part_ids[:remaining])
        return input_ids[:limit]
    def forward(self, original_input: str, expert_outputs: List[Dict[str, Any]], send_to_user: bool = True) -> CentralOutput:
        self.load()
        base_input_ids = self.tokenizer.encode(original_input)[:configs.MAX_SEQ_LEN]
        input_ids = self._build_input_ids(original_input, expert_outputs)
        tokens = mx.array([input_ids])
        t_start = time.perf_counter()
        logits = self.model(tokens)
        mx.eval(logits)
        t_end = time.perf_counter()
        synthesis_hidden = self._compute_hidden_mean(input_ids)
        base_hidden = self._compute_hidden_mean(base_input_ids)
        min_dim = min(base_hidden.shape[0], synthesis_hidden.shape[0])
        contribution_hidden = synthesis_hidden[:min_dim] - base_hidden[:min_dim]
        mx.eval(contribution_hidden)
        last_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]
        token_id = int(mx.argmax(last_logits).item())
        synthesis_text = self.tokenizer.decode([token_id])
        expert_scores = {}
        expert_tkl_scores = {}
        for eo in expert_outputs:
            eid = eo.get("expert_id", 0)
            r_i = self._compute_r_i_from_hidden(eo.get("hidden_states"), contribution_hidden, eo.get("wall_time", 1.0))
            expert_scores[eid] = r_i
        entropy = self.compute_reconstruction_entropy(synthesis_hidden)
        return CentralOutput(
            synthesis_text=synthesis_text,
            synthesis_hidden=synthesis_hidden,
            contribution_hidden=contribution_hidden,
            expert_scores=expert_scores,
            expert_tkl=expert_tkl_scores,
            reconstruction_entropy=entropy,
            send_to_user=send_to_user,
        )
    def compute_r_i(self, expert_output_hidden: mx.array, contribution_hidden: mx.array, wall_time: float) -> float:
        if expert_output_hidden is None or contribution_hidden is None:
            return 0.0
        expert_vec = expert_output_hidden.reshape(-1)
        contribution_vec = contribution_hidden.reshape(-1)
        min_dim = min(int(expert_vec.shape[0]), int(contribution_vec.shape[0]))
        if min_dim <= 0:
            return 0.0
        expert_vec = expert_vec[:min_dim]
        contribution_vec = contribution_vec[:min_dim]
        expert_vec = expert_vec - mx.mean(expert_vec)
        contribution_vec = contribution_vec - mx.mean(contribution_vec)
        eo_norm = mx.linalg.norm(expert_vec)
        contrib_norm = mx.linalg.norm(contribution_vec)
        mx.eval(eo_norm, contrib_norm)
        if float(eo_norm.item()) < 1e-8 or float(contrib_norm.item()) < 1e-8:
            return 0.0
        sim = mx.sum((expert_vec / (eo_norm + 1e-8)) * (contribution_vec / (contrib_norm + 1e-8)))
        mx.eval(sim)
        raw_sim = float(sim.item())
        if not (raw_sim == raw_sim):
            return 0.0
        score = (raw_sim + 1.0) * 0.5
        return max(0.0, min(1.0, score))
    def _compute_r_i_from_hidden(self, expert_hidden, contribution_hidden, wall_time):
        if expert_hidden is None:
            return 0.0
        if isinstance(expert_hidden, mx.array):
            return self.compute_r_i(expert_hidden, contribution_hidden, wall_time)
        return 0.0
    def compute_tkl(self, r_i: float, r_out: float, historical_anchor: float, c_e: float) -> float:
        if c_e < 1e-9:
            c_e = 1e-9
        tkl = r_out * (r_i / c_e) * historical_anchor
        return max(float(configs.TKL_FLOOR), tkl)
    def update_r_t(self, expert_id: int, token_count: int, wall_time: float, convolution: ApexNadirConvolution):
        convolution.update_latency(expert_id, token_count, wall_time)
    def compute_reconstruction_entropy(self, synthesis_hidden: mx.array) -> float:
        if synthesis_hidden is None:
            return 0.0
        values = np.asarray(synthesis_hidden.tolist(), dtype=np.float64).reshape(-1)
        if values.size == 0:
            return 0.0
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return 0.0
        values = values[finite_mask]
        values = np.clip(values, -1e4, 1e4)
        values = values - np.max(values)
        exp_values = np.exp(values)
        denom = float(exp_values.sum())
        if denom <= 0.0 or not np.isfinite(denom):
            return 0.0
        probs = exp_values / denom
        probs = np.clip(probs, 1e-12, 1.0)
        entropy = float(-(probs * np.log(probs)).sum())
        if not np.isfinite(entropy):
            return 0.0
        return entropy
    def generate(self, input_text: str, max_tokens: int = 256) -> str:
        self.load()
        from mlx_lm import generate as mlx_generate
        return mlx_generate(self.model, self.tokenizer, prompt=input_text, max_tokens=max_tokens)
