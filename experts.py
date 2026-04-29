from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import mlx.core as mx
import configs
from apex_nadir_convolution import ApexNadirConvolution
from memory import SessionTracker
from splitter import get_available_ram_mb
@dataclass
class ExpertOutput:
    expert_id: int
    output_text: str
    hidden_states: mx.array
    wall_time: float
    token_count: int
    from_cache: bool
class ExpertPool:
    def __init__(self, convolution: ApexNadirConvolution, session_tracker: SessionTracker):
        self.convolution = convolution
        self.session_tracker = session_tracker
        self.loaded_experts: Dict[int, Any] = {}
        self.loaded_tokenizers: Dict[int, Any] = {}
        self.token_allocation_history: Dict[int, deque] = {
            i: deque(maxlen=configs.TKL_HISTORY_LEN) for i in range(configs.EXPERT_POOL_SIZE)
        }
        self.domain_scores: Dict[int, Dict[str, float]] = {
            i: {} for i in range(configs.EXPERT_POOL_SIZE)
        }
        self.current_domain: Dict[int, str] = {
            i: "general" for i in range(configs.EXPERT_POOL_SIZE)
        }
    def get_available_ram_mb(self) -> float:
        return get_available_ram_mb()
    def load_experts(self, expert_ids: List[int]):
        from mlx_lm import load as mlx_load
        for eid in expert_ids:
            if eid in self.loaded_experts:
                continue
            available = self.get_available_ram_mb()
            if available < configs.EXPERT_RAM_MB:
                raise RuntimeError(f"Cannot load expert {eid}: {available:.1f} MB available, {configs.EXPERT_RAM_MB} MB required.")
            try:
                model, tokenizer = mlx_load(configs.EXPERT_MODEL_ID)
                from mlx_lm.tuner.utils import linear_to_lora_layers
                import mlx.core as mx
                from pathlib import Path
                model.freeze()
                lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
                num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
                linear_to_lora_layers(model, num_layers, lora_config)
                weights_path = Path(configs.CHECKPOINT_DIR) / f"expert_{eid:03d}" / "weights.safetensors"
                if weights_path.exists():
                    model.load_weights(str(weights_path), strict=False)
                model.train()
            except Exception as e:
                print(f"[error] Failed to load expert {eid}: {e}")
                continue
            self.loaded_experts[eid] = model
            self.loaded_tokenizers[eid] = tokenizer
    def unload_experts(self, expert_ids: List[int], keep_buffer: Optional[Set[int]] = None):
        keep = keep_buffer or set()
        import mlx.core as mx
        for eid in expert_ids:
            if eid in keep:
                continue
            if eid in self.loaded_experts:
                del self.loaded_experts[eid]
            if eid in self.loaded_tokenizers:
                del self.loaded_tokenizers[eid]
        mx.clear_cache()
    def save_experts(self, expert_ids: Optional[List[int]] = None):
        from mlx.utils import tree_flatten
        from pathlib import Path
        targets = expert_ids if expert_ids is not None else list(self.loaded_experts.keys())
        for eid in targets:
            model = self.loaded_experts.get(eid)
            if model is None:
                continue
            flat_params = dict(tree_flatten(model.trainable_parameters()))
            checkpoint_dir = Path(configs.CHECKPOINT_DIR) / f"expert_{eid:03d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(checkpoint_dir / "weights.safetensors"), flat_params)
    def expert_forward(self, expert_id: int, fragment_tokens: mx.array) -> ExpertOutput:
        if expert_id not in self.loaded_experts:
            raise RuntimeError(f"Expert {expert_id} not loaded.")
        model = self.loaded_experts[expert_id]
        tokenizer = self.loaded_tokenizers[expert_id]
        input_embeds = fragment_tokens.reshape(1, -1)
        t_start = time.perf_counter()
        logits = model(input_embeds)
        mx.eval(logits)
        t_end = time.perf_counter()
        wall_time = t_end - t_start
        token_ids = logits[0].tolist() if hasattr(logits[0], "tolist") else list(logits[0])
        if isinstance(token_ids[0], list):
            token_ids = [int(max(enumerate(row), key=lambda x: x[1])[0]) for row in token_ids]
        output_text = tokenizer.decode(token_ids)
        if hasattr(model, 'model'):
            hidden_out = model.model(input_embeds)
            mx.eval(hidden_out)
        else:
            hidden_out = logits
        if hidden_out.ndim == 3:
            hidden_mean = mx.mean(hidden_out[0], axis=0)
        elif hidden_out.ndim == 2:
            hidden_mean = mx.mean(hidden_out, axis=0)
        else:
            hidden_mean = hidden_out
        mx.eval(hidden_mean)
        tc = fragment_tokens.shape[0]
        self.record_token_allocation(expert_id, tc)
        return ExpertOutput(
            expert_id=expert_id,
            output_text=output_text,
            hidden_states=hidden_mean,
            wall_time=wall_time,
            token_count=tc,
            from_cache=False,
        )
    def get_masking_rate(self, expert_id: int, domain: str) -> float:
        expert_score = self.domain_scores[expert_id].get(domain, 0.0)
        domain_mean = self.session_tracker.get_domain_mean_score(domain)
        if domain_mean < 1e-9:
            return 1.0
        rate = 1.0 - (expert_score / domain_mean)
        return max(0.0, min(1.0, rate))
    def check_stuck_expert(self, expert_id: int, domain: str, token_count: int, convolution: ApexNadirConvolution) -> bool:
        rate = self.get_masking_rate(expert_id, domain)
        if rate <= configs.MASKING_STUCK_THRESHOLD:
            return False
        exposure = self.session_tracker.get_domain_exposure(expert_id, domain)
        domain_mean_exposure = self.session_tracker.get_domain_mean_exposure(domain)
        if exposure < domain_mean_exposure:
            return False
        return True
    def reassign_expert(self, expert_id: int, new_domain: str):
        old_domain = self.current_domain.get(expert_id, "unknown")
        self.domain_scores[expert_id][new_domain] = 0.0
        self.token_allocation_history[expert_id] = deque(maxlen=configs.TKL_HISTORY_LEN)
        self.current_domain[expert_id] = new_domain
        self.convolution.reset_r_t_curve(expert_id)
        if expert_id in self.loaded_experts:
            del self.loaded_experts[expert_id]
            if expert_id in self.loaded_tokenizers:
                del self.loaded_tokenizers[expert_id]
            mx.clear_cache()
        self.session_tracker.record_migration(expert_id, new_domain)
    def record_token_allocation(self, expert_id: int, token_count: int):
        self.token_allocation_history[expert_id].append(token_count)
    def get_historical_anchor(self, expert_id: int) -> float:
        history = list(self.token_allocation_history[expert_id])
        if len(history) < 2:
            return float(configs.FRAGMENT_MIN)
        t_max = float(max(history))
        t_min = float(max(min(history), 1))
        return math.sqrt(t_max * t_min)
    def update_domain_score(self, expert_id: int, domain: str, r_i: float):
        current = self.domain_scores[expert_id].get(domain, 0.0)
        self.domain_scores[expert_id][domain] = configs.EMA_DECAY * current + (1.0 - configs.EMA_DECAY) * r_i
    def check_starvation_eviction(self, expert_id: int, domain: str) -> bool:
        tkl = self.session_tracker.get_expert_tkl(expert_id)
        domain_mean_tkl = self.session_tracker.get_domain_mean_tkl(domain)
        if domain_mean_tkl < 1e-9:
            return False
        return tkl < domain_mean_tkl * 0.5
    def check_monopoly_overflow(self, expert_id: int) -> bool:
        current_alloc = self.session_tracker.get_current_allocation(expert_id)
        return self.convolution.check_monopoly_ceiling(expert_id, current_alloc)
