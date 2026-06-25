from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import mlx.core as mx
from mlx.utils import tree_flatten
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
    def __init__(self, convolution: ApexNadirConvolution, session_tracker: SessionTracker, max_loaded: int = 6):
        self.convolution = convolution
        self.session_tracker = session_tracker
        self.loaded_experts: Dict[int, Any] = {}
        self.loaded_tokenizers: Dict[int, Any] = {}
        self._load_order: deque = deque()  # LRU tracking: oldest at left
        self._max_loaded = max_loaded  # hard cap on concurrent experts in memory
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
    def _model_is_finite(self, model: Any) -> bool:
        flat = dict(tree_flatten(model.trainable_parameters()))
        if not flat:
            return True
        # Single batched eval instead of per-parameter loop to avoid MLX deadlock
        checks = [mx.all(mx.isfinite(v)) for v in flat.values()]
        all_finite = mx.all(mx.stack(checks))
        mx.eval(all_finite)
        return bool(all_finite.item())
    def _checkpoint_is_finite(self, path: Path) -> bool:
        try:
            from safetensors.numpy import load_file
            import numpy as np
            data = load_file(str(path))
            for value in data.values():
                if not np.isfinite(value).all():
                    return False
            return True
        except Exception:
            return False
    def _drop_checkpoint(self, expert_id: int):
        checkpoint_dir = Path(configs.CHECKPOINT_DIR) / f"expert_{expert_id:03d}"
        weights_path = checkpoint_dir / "weights.safetensors"
        if weights_path.exists():
            weights_path.unlink()
    def _evict_lru(self, protect: Optional[Set[int]] = None) -> bool:
        """Evict the least-recently-used expert to free RAM. Returns True if evicted."""
        protect = protect or set()
        for eid in list(self._load_order):
            if eid in protect or eid not in self.loaded_experts:
                continue
            del self.loaded_experts[eid]
            if eid in self.loaded_tokenizers:
                del self.loaded_tokenizers[eid]
            self._load_order.remove(eid)
            mx.clear_cache()
            return True
        return False

    def _touch_lru(self, eid: int):
        """Mark expert as recently used (move to right/end of deque)."""
        if eid in self._load_order:
            self._load_order.remove(eid)
        self._load_order.append(eid)

    def load_experts(self, expert_ids: List[int]):
        from mlx_lm import load as mlx_load
        needed_set = set(expert_ids)
        for eid in expert_ids:
            if eid in self.loaded_experts:
                self._touch_lru(eid)
                continue
            # Evict LRU experts until we're under the hard cap
            while len(self.loaded_experts) >= self._max_loaded:
                if not self._evict_lru(protect=needed_set):
                    break  # can't evict — load what fits, skip the rest
            if len(self.loaded_experts) >= self._max_loaded:
                break  # cache full, continue with what we have
            available = self.get_available_ram_mb()
            if available < configs.EXPERT_RAM_MB:
                if not self._evict_lru(protect=needed_set):
                    break  # no RAM — continue with what we have
            try:
                model, tokenizer = mlx_load(configs.EXPERT_MODEL_ID)
                from mlx_lm.tuner.utils import linear_to_lora_layers
                model.freeze()
                lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
                num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
                linear_to_lora_layers(model, num_layers, lora_config)
                weights_path = Path(configs.CHECKPOINT_DIR) / f"expert_{eid:03d}" / "weights.safetensors"
                if weights_path.exists() and not self._checkpoint_is_finite(weights_path):
                    print(f"[warn] Dropping corrupt expert checkpoint {eid}")
                    self._drop_checkpoint(eid)
                if weights_path.exists():
                    model.load_weights(str(weights_path), strict=False)
                model.train()
                if not self._model_is_finite(model):
                    print(f"[warn] Expert {eid} loaded non-finite weights, resetting to base")
                    self._drop_checkpoint(eid)
                    model, tokenizer = mlx_load(configs.EXPERT_MODEL_ID)
                    model.freeze()
                    linear_to_lora_layers(model, num_layers, lora_config)
                    model.train()
            except Exception as e:
                print(f"[error] Failed to load expert {eid}: {e}")
                continue
            self.loaded_experts[eid] = model
            self.loaded_tokenizers[eid] = tokenizer
            self._touch_lru(eid)
    def unload_experts(self, expert_ids: List[int], keep_buffer: Optional[Set[int]] = None):
        keep = keep_buffer or set()
        for eid in expert_ids:
            if eid in keep:
                continue
            if eid in self.loaded_experts:
                del self.loaded_experts[eid]
            if eid in self.loaded_tokenizers:
                del self.loaded_tokenizers[eid]
        mx.clear_cache()
    def save_experts(self, expert_ids: Optional[List[int]] = None):
        targets = expert_ids if expert_ids is not None else list(self.loaded_experts.keys())
        for eid in targets:
            model = self.loaded_experts.get(eid)
            if model is None:
                continue
            # Skip the expensive _model_is_finite check during save.
            # Load-time validation (_checkpoint_is_finite) catches corrupt files.
            flat_params = dict(tree_flatten(model.trainable_parameters()))
            checkpoint_dir = Path(configs.CHECKPOINT_DIR) / f"expert_{eid:03d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(checkpoint_dir / "weights.safetensors"), flat_params)
    def _generate_expert_text(self, model: Any, tokenizer: Any, fragment_tokens: mx.array) -> str:
        """Produce a REAL generated answer from the expert (audit A.2.4).

        The old path took argmax(logits) over the expert's own input positions —
        a teacher-forced, per-position next-token prediction, i.e. a scrambled
        echo of the question rather than a response. This generates an actual
        continuation that can be injected into Central's prompt so the expert
        pool reaches the user-facing reply. Greedy (no sampler) for determinism,
        matching CentralModel.generate. Only called when the text will be shown."""
        try:
            from mlx_lm import generate as mlx_generate
            prompt_text = tokenizer.decode(fragment_tokens.reshape(-1).tolist())
            if not prompt_text.strip():
                return ""
            # Disable LoRA dropout during generation for stable, repeatable text.
            was_training = bool(getattr(model, "training", False))
            if was_training:
                model.eval()
            try:
                text = mlx_generate(
                    model, tokenizer, prompt=prompt_text,
                    max_tokens=configs.EXPERT_GEN_MAX_TOKENS,
                )
            finally:
                if was_training:
                    model.train()
            return text.strip()
        except Exception as e:
            print(f"[warn] expert text generation failed: {e}")
            return ""

    def _argmax_text(self, logits: mx.array, tokenizer: Any) -> str:
        """Cheap, no-generation expert text for the training/measurement path.

        Not user-facing — it only perturbs Central's synthesis input so the r_i
        contribution delta (synthesis_hidden - base_hidden) stays non-trivial.
        Kept argmax-cheap to preserve training throughput; the real reply path
        uses _generate_expert_text instead."""
        if logits.ndim == 3:
            token_ids = mx.argmax(logits[0], axis=-1).tolist()
        elif logits.ndim == 2:
            token_ids = mx.argmax(logits, axis=-1).tolist()
        else:
            token_ids = [int(mx.argmax(logits).item())]
        return tokenizer.decode(token_ids)

    def expert_forward(self, expert_id: int, fragment_tokens: mx.array, generate_text: bool = False) -> ExpertOutput:
        if expert_id not in self.loaded_experts:
            raise RuntimeError(f"Expert {expert_id} not loaded.")
        model = self.loaded_experts[expert_id]
        tokenizer = self.loaded_tokenizers[expert_id]
        input_embeds = fragment_tokens.reshape(1, -1)
        t_start = time.perf_counter()
        # Single backbone pass: get hidden states, then derive logits cheaply
        if hasattr(model, 'model'):
            hidden_out = model.model(input_embeds)
            mx.eval(hidden_out)
            # Derive logits from hidden states (just the lm_head, no second pass)
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(hidden_out)
            else:
                logits = model(input_embeds)
            mx.eval(logits)
        else:
            logits = model(input_embeds)
            mx.eval(logits)
            hidden_out = logits
        t_end = time.perf_counter()
        wall_time = t_end - t_start
        # generate_text=True (Timeline B reply): a real expert answer that reaches
        # Central's generation prompt. generate_text=False (training/measurement):
        # the cheap argmax echo, used only to perturb Central's synthesis input for
        # r_i scoring — never shown to the user. See audit A.2.4.
        if generate_text:
            output_text = self._generate_expert_text(model, tokenizer, fragment_tokens)
        else:
            output_text = self._argmax_text(logits, tokenizer)
        # Hidden states already computed above (no extra pass)
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
        # Require minimum activations in new domain before eviction is eligible.
        # Prevents the death spiral: migrate → bad first batch → migrate again → r_i stays 0.
        if self.session_tracker.get_expert_activations(expert_id) < configs.STARVATION_MIN_ACTIVATIONS:
            return False
        tkl = self.session_tracker.get_expert_tkl(expert_id)
        domain_mean_tkl = self.session_tracker.get_domain_mean_tkl(domain)
        if domain_mean_tkl < 1e-9:
            return False
        return tkl < domain_mean_tkl * 0.5
    def check_monopoly_overflow(self, expert_id: int) -> bool:
        current_alloc = self.session_tracker.get_current_allocation(expert_id)
        return self.convolution.check_monopoly_ceiling(expert_id, current_alloc)
