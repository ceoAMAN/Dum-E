from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Set
import mlx.core as mx
import numpy as np
import configs
from apex_nadir_convolution import ApexNadirConvolution
@dataclass
class DomainTopography:
    domain_map: Dict[int, str]
    domain_proportions: Dict[str, float]
    total_tokens: int
@dataclass
class GateOutput:
    hidden_states: mx.array
    k_per_token: int
    domain_logits: mx.array
    timeline_flag: str
    confidence: float
@dataclass
class SelectedExpert:
    expert_id: int
    distance_to_peak: float
    domain: str
    is_alpha: bool
class GateModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
    def load(self):
        if self._loaded:
            return
        from mlx_lm import load as mlx_load
        import mlx.core as mx
        from pathlib import Path
        self.model, self.tokenizer = mlx_load(configs.GATE_MODEL_ID)
        from mlx_lm.tuner.utils import linear_to_lora_layers
        self.model.freeze()
        lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
        num_layers = len(self.model.layers) if hasattr(self.model, "layers") else len(self.model.model.layers)
        linear_to_lora_layers(self.model, num_layers, lora_config)
        weights_path = Path(configs.CHECKPOINT_DIR) / "gate" / "weights.safetensors"
        if weights_path.exists():
            self.model.load_weights(str(weights_path), strict=False)
        self.model.train()
        self._loaded = True
    def look_ahead(self, tokens: mx.array) -> DomainTopography:
        self.load()
        domain_map: Dict[int, str] = {}
        total = tokens.shape[0]
        if total == 0:
            return DomainTopography(domain_map={}, domain_proportions={}, total_tokens=0)
        # Use backbone to get actual hidden states, not logits
        if hasattr(self.model, 'model'):
            hidden = self.model.model(tokens.reshape(1, -1))
        else:
            hidden = self.model(tokens.reshape(1, -1))
        mx.eval(hidden)
        domain_counts: Dict[str, int] = {}
        chunk_size = max(1, total // 10)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_hidden = hidden[0, start:end, :]
            mean_hidden = mx.mean(chunk_hidden, axis=0)
            mx.eval(mean_hidden)
            vals = mean_hidden.tolist()
            domain = self._classify_domain(vals)
            for idx in range(start, end):
                domain_map[idx] = domain
            domain_counts[domain] = domain_counts.get(domain, 0) + (end - start)
        domain_proportions = {d: c / total for d, c in domain_counts.items()}
        return DomainTopography(domain_map=domain_map, domain_proportions=domain_proportions, total_tokens=total)
    def _classify_domain(self, hidden_values: list) -> str:
        if not hidden_values:
            return "general"
        variance = float(np.var(hidden_values[:64]))
        mean_abs = float(np.mean(np.abs(hidden_values[:64])))
        if variance > 2.0:
            return "code"
        if mean_abs > 1.5:
            return "reasoning"
        if mean_abs < 0.3:
            return "knowledge"
        return "general"
    def forward(self, tokens: mx.array) -> GateOutput:
        self.load()
        # Use backbone (model.model) to get actual hidden states, not logits
        if hasattr(self.model, 'model'):
            hidden = self.model.model(tokens.reshape(1, -1))
        else:
            hidden = self.model(tokens.reshape(1, -1))
        mx.eval(hidden)
        mean_hidden = mx.mean(hidden[0], axis=0)
        mean_hidden = mx.clip(mean_hidden, -1e4, 1e4)
        mx.eval(mean_hidden)
        domain_logits = mean_hidden[:configs.K_MAX]
        probs = mx.softmax(domain_logits)
        mx.eval(probs)
        entropy = -float(mx.sum(probs * mx.log(probs + 1e-10)).item())
        max_entropy = math.log(max(configs.K_MAX, 2))
        confidence = max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))
        if confidence > configs.FAST_PATH_THRESHOLD:
            timeline = "A"
        else:
            timeline = "B"
        k = max(configs.K_MIN, min(configs.K_MAX, int((1.0 - confidence) * configs.K_MAX)))
        return GateOutput(hidden_states=mean_hidden, k_per_token=k, domain_logits=domain_logits, timeline_flag=timeline, confidence=confidence)
    def parameters(self) -> dict:
        if self.model is not None:
            return self.model.parameters()
        return {}
class MaskingSchedule:
    def __init__(self):
        self._last_masked: Set[int] = set()
    def get_masked_experts(self, alpha_experts: List[int], batch_id: int) -> Set[int]:
        if not alpha_experts:
            return set()
        candidates = [e for e in alpha_experts if e not in self._last_masked]
        if not candidates:
            candidates = alpha_experts
        mask_count = max(1, len(candidates) // 3)
        masked = set(candidates[:mask_count])
        self._last_masked = masked
        return masked
class TripleKSelector:
    def __init__(self, convolution: ApexNadirConvolution):
        self.convolution = convolution
        self.k_d: Dict[str, List[int]] = {}
        self.k_pd: Dict[str, List[int]] = {}
        self.k_all: List[int] = list(range(configs.EXPERT_POOL_SIZE))
        self.alpha_experts: Dict[str, List[int]] = {}
        self.beta_experts: Dict[str, List[int]] = {}
        self._rng = random.Random(42)
        self._auto_seed()
    def _auto_seed(self):
        if hasattr(configs, "EXPERT_GROUPS") and configs.EXPERT_GROUPS:
            self.seed_from_calibration(configs.EXPERT_GROUPS)
    def seed_from_calibration(self, domain_assignments: Dict[str, List[int]]):
        self.k_d = dict(domain_assignments)
        for domain, experts in domain_assignments.items():
            n_alpha = max(1, len(experts) // 3)
            self.alpha_experts[domain] = experts[:n_alpha]
            self.beta_experts[domain] = experts[n_alpha:]
            self.k_pd[domain] = list(experts)
    def select_experts(self, gate_output: GateOutput, session_tracker, masking_schedule: MaskingSchedule, batch_id: int = 0, loaded_experts=None) -> List[SelectedExpert]:
        domain = "general"
        logits = gate_output.domain_logits
        if logits is not None and logits.shape[0] > 0:
            vals = logits.tolist()
            domains = ["code", "reasoning", "knowledge", "general"]
            if len(vals) >= len(domains):
                best_idx = int(np.argmax(vals[:len(domains)]))
                domain = domains[best_idx]
        k = gate_output.k_per_token
        domain_pool = self.k_d.get(domain, self.k_all)
        masked = masking_schedule.get_masked_experts(self.alpha_experts.get(domain, []), batch_id)
        candidates = []
        for eid in domain_pool:
            if eid in masked:
                continue
            current_alloc = session_tracker.get_current_allocation(eid)
            dist = self.convolution.get_distance_to_peak(eid, current_alloc)
            if loaded_experts and eid in loaded_experts:
                dist *= 0.1  # Huge discount for RAM-resident experts to avoid SSD latency
            jitter = self._rng.random() * 1e-6
            is_alpha = eid in self.alpha_experts.get(domain, [])
            candidates.append(SelectedExpert(expert_id=eid, distance_to_peak=dist + jitter, domain=domain, is_alpha=is_alpha))
        self._rng.shuffle(candidates)
        candidates.sort(key=lambda e: e.distance_to_peak)
        selected = candidates[:k]
        if len(selected) < k:
            for eid in self.k_all:
                if len(selected) >= k:
                    break
                if any(s.expert_id == eid for s in selected):
                    continue
                if eid in masked:
                    continue
                current_alloc = session_tracker.get_current_allocation(eid)
                dist = self.convolution.get_distance_to_peak(eid, current_alloc)
                selected.append(SelectedExpert(expert_id=eid, distance_to_peak=dist, domain=domain, is_alpha=False))
        return selected[:k]
