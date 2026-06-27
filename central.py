from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
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
        # Default reply length (callers may override for shorter outputs).
        self.gen_max_tokens = 256
    def load(self):
        if self._loaded:
            return
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from pathlib import Path
        self.model, self.tokenizer = mlx_load(configs.CENTRAL_MODEL_ID)
        self.model.freeze()
        lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
        num_layers = len(self.model.layers) if hasattr(self.model, "layers") else len(self.model.model.layers)
        linear_to_lora_layers(self.model, num_layers, lora_config)
        weights_path = Path(configs.CHECKPOINT_DIR) / "central" / "weights.safetensors"
        if weights_path.exists():
            self.model.load_weights(str(weights_path), strict=False)
        self.model.train()
        self._loaded = True
    def _build_input_ids(self, original_input: str, expert_outputs: List[Dict[str, Any]]):
        """Returns (input_ids, n_question) where n_question is the number of
        leading tokens that are the original question (before any expert output
        is appended). Because attention is causal, the synthesis backbone's
        hidden states at positions [0:n_question) equal what a question-only
        forward would produce — so the caller can recover base_hidden from the
        synthesis pass and skip a second 7B forward."""
        limit = configs.MAX_SEQ_LEN
        reserve = 128
        base_limit = max(configs.FRAGMENT_MIN, limit - reserve)
        input_ids = self.tokenizer.encode(original_input)[:base_limit]
        n_question = len(input_ids)
        if len(input_ids) >= limit:
            return input_ids[:limit], min(n_question, limit)
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
        return input_ids[:limit], n_question
    def forward(self, original_input: str, expert_outputs: List[Dict[str, Any]], send_to_user: bool = True) -> CentralOutput:
        self.load()
        input_ids, n_question = self._build_input_ids(original_input, expert_outputs)
        tokens = mx.array([input_ids])
        # ONE backbone pass over [question | expert outputs].
        has_backbone = hasattr(self.model, 'model')
        if has_backbone:
            hidden_out = self.model.model(tokens)
        else:
            hidden_out = self.model(tokens)
        seq_hidden = hidden_out[0] if hidden_out.ndim == 3 else hidden_out  # (T, D)
        # Synthesis = mean over the whole sequence (question + expert context).
        synthesis_hidden = mx.mean(seq_hidden, axis=0)
        # Base = mean over JUST the question prefix of the SAME pass. Causal
        # attention means those positions never saw the expert tokens, so this is
        # identical to a separate question-only forward — but free. Eliminates the
        # second 7B backbone pass that used to dominate per-batch cost.
        n_q = max(1, min(int(n_question), int(seq_hidden.shape[0])))
        base_hidden = mx.mean(seq_hidden[:n_q], axis=0)
        min_dim = min(base_hidden.shape[0], synthesis_hidden.shape[0])
        contribution_hidden = synthesis_hidden[:min_dim] - base_hidden[:min_dim]
        # Single sync for everything training needs (lets MLX fuse/pipeline the rest).
        mx.eval(synthesis_hidden, contribution_hidden)
        # The lm_head vocab projection (512 x 32k matmul) + argmax + decode exist ONLY
        # to produce synthesis_text for the user reply. Training never reads it, so
        # skip the whole thing unless we're actually replying — a free per-batch win.
        if send_to_user:
            if has_backbone and hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(hidden_out)
            else:
                logits = self.model(tokens)
            mx.eval(logits)
            last_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]
            token_id = int(mx.argmax(last_logits).item())
            synthesis_text = self.tokenizer.decode([token_id])
        else:
            synthesis_text = ""
        # expert_scores used to be computed here (one compute_r_i per expert, each
        # forcing 2 mx.eval host-syncs + 3 .item() device->host pulls). Nothing
        # ever read CentralOutput.expert_scores — the training loop recomputes the
        # identical r_i at finetune.py via central.compute_r_i(eo.hidden_states,
        # central_out.contribution_hidden, ...) from the same inputs. So this loop
        # was a per-expert-per-batch duplicate of work done later; dropped. Kept
        # the field as an empty dict for backward-compatible callers.
        expert_scores: Dict[int, float] = {}
        expert_tkl_scores: Dict[int, float] = {}
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
    def _cosine_terms(self, a: mx.array, b: mx.array):
        """Mean-centred cosine of two vectors, returned UNEVALUATED as
        (sim, min_norm) mx scalars so callers can batch the host-sync. sim is raw
        cosine in [-1, 1]; min_norm lets the caller reject degenerate (~zero) vecs."""
        a = a.reshape(-1)
        b = b.reshape(-1)
        m = min(int(a.shape[0]), int(b.shape[0]))
        a = a[:m] - mx.mean(a[:m])
        b = b[:m] - mx.mean(b[:m])
        a_norm = mx.linalg.norm(a)
        b_norm = mx.linalg.norm(b)
        sim = mx.sum((a / (a_norm + 1e-8)) * (b / (b_norm + 1e-8)))
        return sim, mx.minimum(a_norm, b_norm)
    def compute_r_i(
        self,
        expert_output_hidden: mx.array,
        contribution_hidden: mx.array,
        wall_time: float,
        synthesis_hidden: Optional[mx.array] = None,
    ) -> float:
        """Expert contribution score in [0, 1] — alignment only (speed is folded in
        later by compute_tkl via /C_e, so it must NOT be double-counted here).

        Two complementary signals (audit: 'both are needed'):
          - direction:     cosine(expert, contribution_hidden) — did the expert push
                           Central in the direction it actually moved?
          - compatibility: cosine(expert, synthesis_hidden)    — does the expert
                           agree with Central's final synthesised view?
        With no synthesis_hidden it degrades to the direction signal alone (keeps
        older callers working). One mx.eval + one host pull for the whole score."""
        if expert_output_hidden is None or contribution_hidden is None:
            return 0.0
        dir_sim, dir_norm = self._cosine_terms(expert_output_hidden, contribution_hidden)
        if synthesis_hidden is not None:
            comp_sim, comp_norm = self._cosine_terms(expert_output_hidden, synthesis_hidden)
            combined = 0.5 * (dir_sim + 1.0) * 0.5 + 0.5 * (comp_sim + 1.0) * 0.5
            min_norm = mx.minimum(dir_norm, comp_norm)
        else:
            combined = (dir_sim + 1.0) * 0.5
            min_norm = dir_norm
        packed = mx.stack([combined, min_norm])
        mx.eval(packed)
        score, norm = (float(x) for x in packed.tolist())
        if norm < 1e-8 or score != score:   # degenerate vector or NaN
            return 0.0
        return max(0.0, min(1.0, score))
    def compute_l_eff(
        self,
        expert_output_hidden: mx.array,
        synthesis_hidden: mx.array,
        token_count: int,
        wall_time: float,
    ) -> float:
        """Raw efficiency score for one expert (the gate's routing head learns to
        prefer high values). Per the manual: synthesis_compatibility + throughput.
        Returned un-normalised; the caller L1-normalises across the active experts
        before building the gate's L_eff target. Higher = more useful per second."""
        if expert_output_hidden is None or synthesis_hidden is None:
            return 0.0
        sim, min_norm = self._cosine_terms(expert_output_hidden, synthesis_hidden)
        packed = mx.stack([sim, min_norm])
        mx.eval(packed)
        raw_sim, norm = (float(x) for x in packed.tolist())
        if norm < 1e-8 or raw_sim != raw_sim:
            return 0.0
        compatibility = (raw_sim + 1.0) * 0.5                  # [0, 1]
        throughput = token_count / max(wall_time, configs.L_EFF_EPS)
        return max(0.0, compatibility + throughput)
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
        values = np.asarray(synthesis_hidden, dtype=np.float64).reshape(-1)
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
    def format_prompt(self, input_text: str, expert_context: Optional[List[str]] = None) -> str:
        # Audit A.2.4: when the expert pool actually ran (Timeline B), inject the
        # experts' generated analyses into Central's generation prompt so the MoE
        # machinery reaches the deployed reply instead of being a training-only
        # apparatus. With no expert context this is the plain question prompt, so
        # Timeline A and the no-expert fallbacks are unchanged.
        if expert_context:
            notes = "\n".join(f"- {c.strip()}" for c in expert_context if c and c.strip())
            if notes:
                return (
                    f"<s> [INST] {input_text}\n\n"
                    f"Expert analyses to consider:\n{notes}\n\n"
                    f"Use the analyses where they help and give the best final answer. [/INST]"
                )
        return f"<s> [INST] {input_text} [/INST]"
    def generate(self, input_text: str, max_tokens: Optional[int] = None,
                 expert_context: Optional[List[str]] = None) -> str:
        self.load()
        from mlx_lm import generate as mlx_generate
        mt = max_tokens if max_tokens is not None else self.gen_max_tokens
        prompt = self.format_prompt(input_text, expert_context)
        return mlx_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=mt)

    def generate_stream(self, input_text: str, max_tokens: Optional[int] = None):
        """Yield text deltas as they are generated (for low-latency speech)."""
        self.load()
        from mlx_lm import stream_generate
        mt = max_tokens if max_tokens is not None else self.gen_max_tokens
        for response in stream_generate(
            self.model, self.tokenizer,
            prompt=self.format_prompt(input_text), max_tokens=mt,
        ):
            text = getattr(response, "text", "")
            if text:
                yield text
