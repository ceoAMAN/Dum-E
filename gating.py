from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import mlx.core as mx
import mlx.nn as nn
import configs
from apex_nadir_convolution import ApexNadirConvolution

# Domain slots, in the order the gate's first 4 domain-logit dims map to and the
# L_dom one-hot target uses. Single source of truth shared by routing + topography.
DOMAINS = ["code", "reasoning", "knowledge", "general"]
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
    # Learned per-expert routing preference from GateNet.route_head (shape
    # [EXPERT_POOL_SIZE]). None only on the NaN-guard fallback path. select_experts
    # blends this with apex-nadir distance-to-peak.
    route_logits: Optional[mx.array] = None
@dataclass
class SelectedExpert:
    expert_id: int
    distance_to_peak: float
    domain: str
    is_alpha: bool
class GateNet(nn.Module):
    """The trainable gate as one unit, so a single value_and_grad call updates the
    backbone's LoRA params and the expert-routing head together. `backbone` is the
    LoRA'd Qwen (only its LoRA adapters are unfrozen); `route_head` maps the pooled
    gate hidden state to one preference logit per expert — the differentiable path
    that L_eff/L_rel use to teach the gate which experts to prefer."""
    def __init__(self, backbone: nn.Module, d_model: int, n_experts: int):
        super().__init__()
        self.backbone = backbone
        self.route_head = nn.Linear(d_model, n_experts)

class GateModel:
    def __init__(self):
        self.model = None        # LoRA'd Qwen backbone (also held by self.net.backbone)
        self.tokenizer = None
        self.net = None          # GateNet: backbone + route_head, the trainable unit
        self._loaded = False
    @property
    def route_head(self):
        return self.net.route_head if self.net is not None else None
    def _route_head_path(self) -> Path:
        return Path(configs.CHECKPOINT_DIR) / "gate" / "route_head.safetensors"
    def load(self):
        if self._loaded:
            return
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        self.model, self.tokenizer = mlx_load(configs.GATE_MODEL_ID)
        self.model.freeze()
        lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
        num_layers = len(self.model.layers) if hasattr(self.model, "layers") else len(self.model.model.layers)
        linear_to_lora_layers(self.model, num_layers, lora_config)
        weights_path = Path(configs.CHECKPOINT_DIR) / "gate" / "weights.safetensors"
        if weights_path.exists():
            self.model.load_weights(str(weights_path), strict=False)
        # Wrap backbone + a fresh routing head into the trainable unit. The head's
        # own checkpoint is restored separately so it survives restarts.
        self.net = GateNet(self.model, configs.GATE_D_MODEL, configs.EXPERT_POOL_SIZE)
        rh_path = self._route_head_path()
        if rh_path.exists():
            self.net.route_head.load_weights(str(rh_path), strict=False)
        mx.eval(self.net.route_head.parameters())
        self.model.train()
        self._loaded = True
    def save_route_head(self):
        from mlx.utils import tree_flatten
        p = self._route_head_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(p), dict(tree_flatten(self.net.route_head.parameters())))
    def _backbone(self, tokens: mx.array) -> mx.array:
        """One full backbone pass → per-token hidden states (1, T, D). Shared by
        forward(), look_ahead() and forward_with_topography() so a single request
        never runs the backbone twice."""
        self.load()
        if hasattr(self.model, 'model'):
            hidden = self.model.model(tokens.reshape(1, -1))
        else:
            hidden = self.model(tokens.reshape(1, -1))
        mx.eval(hidden)
        return hidden
    def _topography_from_hidden(self, hidden: mx.array, total: int) -> DomainTopography:
        domain_map: Dict[int, str] = {}
        if total == 0:
            return DomainTopography(domain_map={}, domain_proportions={}, total_tokens=0)
        domain_counts: Dict[str, int] = {}
        chunk_size = max(1, total // 10)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_hidden = hidden[0, start:end, :]
            mean_hidden = mx.mean(chunk_hidden, axis=0)
            domain = self._domain_from_mean_hidden(mean_hidden)
            for idx in range(start, end):
                domain_map[idx] = domain
            domain_counts[domain] = domain_counts.get(domain, 0) + (end - start)
        domain_proportions = {d: c / total for d, c in domain_counts.items()}
        return DomainTopography(domain_map=domain_map, domain_proportions=domain_proportions, total_tokens=total)
    def look_ahead(self, tokens: mx.array) -> DomainTopography:
        total = int(tokens.shape[0])
        if total == 0:
            return DomainTopography(domain_map={}, domain_proportions={}, total_tokens=0)
        return self._topography_from_hidden(self._backbone(tokens), total)
    def forward_with_topography(self, tokens: mx.array):
        """Fused: ONE backbone pass → (GateOutput, DomainTopography). Timeline B
        previously paid for two full gate passes per request (forward() then
        look_ahead()); this collapses them into one — the single biggest routing
        overhead on the inference path."""
        hidden = self._backbone(tokens)
        gate_out = self.forward(tokens, hidden=hidden)
        topo = self._topography_from_hidden(hidden, int(tokens.shape[0]))
        return gate_out, topo
    def _domain_from_mean_hidden(self, mean_hidden: mx.array) -> str:
        """Domain of a (chunk) mean hidden state from the gate's LEARNED domain head:
        the same z-score → domain_logits[:4] argmax that forward() uses for routing.
        No hardcoded variance/magnitude thresholds — the trained gate decides."""
        h = mx.clip(mean_hidden, -1e4, 1e4)
        mu = mx.mean(h)
        sigma = mx.sqrt(mx.mean((h - mu) ** 2) + 1e-8)
        domain_logits = ((h - mu) / (sigma + 1e-8))[:len(DOMAINS)]
        return DOMAINS[int(mx.argmax(domain_logits).item())]
    def forward(self, tokens: mx.array, hidden: mx.array = None) -> GateOutput:
        self.load()
        # Reuse a precomputed backbone pass when the caller already ran one
        # (forward_with_topography). Otherwise run it here. Either way the gate
        # backbone executes exactly once per request.
        if hidden is None:
            hidden = self._backbone(tokens)
        mean_hidden = mx.mean(hidden[0], axis=0)
        mean_hidden = mx.clip(mean_hidden, -1e4, 1e4)
        mx.eval(mean_hidden)
        # Guard against NaN hidden states (corrupt checkpoint or exploding backbone).
        import math as _math
        spot_vals = mean_hidden[:8].tolist()
        if any(not _math.isfinite(v) for v in spot_vals):
            domain_logits = mx.zeros(4)
            k = configs.K_DEFAULT
            return GateOutput(
                hidden_states=mean_hidden,
                k_per_token=k,
                domain_logits=domain_logits,
                timeline_flag="B",
                confidence=0.0,
            )
        # z-score normalise across ALL hidden dims before slicing domain logits.
        # Raw backbone activations have large magnitudes (e.g. ±100) → softmax
        # saturates → confidence locks at 1.0 before any training. After
        # normalisation values are ~N(0,1), entropy is meaningful, and the gate
        # can actually learn to discriminate domains via l_dom gradients.
        mu = mx.mean(mean_hidden)
        sigma = mx.sqrt(mx.mean((mean_hidden - mu) ** 2) + 1e-8)
        normed = (mean_hidden - mu) / (sigma + 1e-8)
        domain_logits = normed[:4]   # 4 slots: code / reasoning / knowledge / general
        # Learned per-expert routing preference from the same pooled hidden state.
        route_logits = self.net.route_head(mean_hidden)
        mx.eval(domain_logits, route_logits)
        probs = mx.softmax(domain_logits)
        mx.eval(probs)
        entropy = -float(mx.sum(probs * mx.log(probs + 1e-10)).item())
        max_entropy = math.log(4)    # 4 domain classes
        confidence = max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))
        if confidence > configs.FAST_PATH_THRESHOLD:
            timeline = "A"
        else:
            timeline = "B"
        k = max(configs.K_MIN, min(configs.K_MAX, int((1.0 - confidence) * configs.K_MAX)))
        return GateOutput(hidden_states=mean_hidden, k_per_token=k, domain_logits=domain_logits, timeline_flag=timeline, confidence=confidence, route_logits=route_logits)
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
        if logits is not None and logits.shape[0] >= len(DOMAINS):
            domain = DOMAINS[int(mx.argmax(logits[:len(DOMAINS)]).item())]
        # k=0 means "Timeline A" (no experts) — but that path never calls
        # select_experts. Whenever we ARE selecting (Timeline B / training), we need
        # at least one expert; otherwise selected[:0] == [] and the caller skips the
        # batch entirely (no gate/expert gradient). A confident gate emits k=0, which
        # was silently zeroing ~75% of training batches.
        k = max(1, gate_output.k_per_token)
        domain_pool = self.k_d.get(domain, self.k_all)
        masked = masking_schedule.get_masked_experts(self.alpha_experts.get(domain, []), batch_id)
        # The gate's learned routing preference (one softmax pull). Blended into the
        # ranking below: apex-nadir distance-to-peak keeps things grounded, the head
        # tilts selection toward experts it has learned are efficient/fresh.
        route_pref = None
        if gate_output.route_logits is not None:
            route_pref = mx.softmax(gate_output.route_logits).tolist()
        candidates = []
        for eid in domain_pool:
            if eid in masked:
                continue
            current_alloc = session_tracker.get_current_allocation(eid)
            dist = self.convolution.get_distance_to_peak(eid, current_alloc)
            if loaded_experts and eid in loaded_experts:
                dist *= 0.1  # Huge discount for RAM-resident experts to avoid SSD latency
            if route_pref is not None and eid < len(route_pref):
                dist -= configs.ROUTE_BIAS_W * route_pref[eid]   # learned preference lowers rank-distance
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
