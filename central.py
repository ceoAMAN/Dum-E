from __future__ import annotations
import math
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
        self._context_limit: Optional[int] = None
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
        lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA / configs.LORA_R, "dropout": configs.LORA_DROPOUT}
        num_layers = len(self.model.layers) if hasattr(self.model, "layers") else len(self.model.model.layers)
        linear_to_lora_layers(self.model, num_layers, lora_config)
        weights_path = Path(configs.CHECKPOINT_DIR) / "central" / "weights.safetensors"
        if weights_path.exists():
            self.model.load_weights(str(weights_path), strict=False)
        self.model.train()
        self._loaded = True
    def save(self):
        """Persist Central's trained LoRA adapters. Counterpart to
        ExpertPool.save_experts / GateModel.save_route_head, which existed while
        this did not — so even a correct training loop could not keep Central's
        weights, and every boot silently reloaded the stock model (load() guards
        on `if weights_path.exists()`, which was never true)."""
        from mlx.utils import tree_flatten
        from pathlib import Path
        if self.model is None:
            return
        out_dir = Path(configs.CHECKPOINT_DIR) / "central"
        out_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(out_dir / "weights.safetensors"),
            dict(tree_flatten(self.model.trainable_parameters())),
        )
    def context_limit(self) -> int:
        """Central's REAL context window, read from the model it is actually
        running — not an invented constant. MAX_SEQ_LEN survives only as a
        fallback for a model that reports nothing."""
        if self._context_limit is not None:
            return self._context_limit
        self.load()
        limit = None
        args = getattr(self.model, "args", None)
        for attr in ("max_position_embeddings", "max_seq_len", "context_length"):
            v = getattr(args, attr, None)
            if isinstance(v, int) and v > 0:
                limit = v
                break
        if limit is None:
            v = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(v, int) and 0 < v < 10 ** 7:
                limit = v
        self._context_limit = int(limit or configs.MAX_SEQ_LEN)
        return self._context_limit
    def _build_input_ids(self, original_input: str, expert_outputs: List[Dict[str, Any]], limit: Optional[int] = None):
        """Returns (input_ids, n_question) where n_question is the number of
        leading tokens that are the original question (before any expert output
        is appended). Because attention is causal, the synthesis backbone's
        hidden states at positions [0:n_question) equal what a question-only
        forward would produce — so the caller can recover base_hidden from the
        synthesis pass and skip a second forward.

        Central sees the WHOLE input and EVERY expert output. There is no reserve
        and no question truncation: the question is never shortened to make room
        for expert context. The old fixed `reserve = 128` against a 128-token
        window cut every question down to FRAGMENT_MIN (32) tokens, so Central
        scored every expert, computed every contribution vector and drove every
        migration decision while seeing a stub of the input. Capping at an
        invented constant was the mistake, not the size of the constant — the only
        real bound is the model's own context window."""
        limit = limit if limit is not None else self.context_limit()
        input_ids = self.tokenizer.encode(original_input)
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
    def build_training_ids(self, original_input: str, expert_outputs: List[Dict[str, Any]], target_ids: List[int]):
        """Prompt+expert context (same construction as _build_input_ids) followed
        by REAL target tokens — e.g. the actual continuation of a real corpus
        sample — for teacher-forced grounding.

        Returns (input_ids, n_context): n_context is where the target region
        starts, so the caller (training.compute_central_ce) can mask the
        cross-entropy to target positions only — the model is never scored on
        predicting the prompt or expert context, only the real continuation.

        The ONLY reservation here is for the targets themselves, which is real:
        without room for the continuation there is nothing grounded to learn
        from. The question and the expert outputs are never trimmed to make room
        for each other."""
        limit = self.context_limit()
        target_ids = list(target_ids)
        target_budget = min(len(target_ids), max(1, limit // 4)) if target_ids else 0
        ctx_cap = max(1, limit - target_budget)
        context_ids, _ = self._build_input_ids(original_input, expert_outputs, limit=ctx_cap)
        remaining = max(0, limit - len(context_ids))
        input_ids = context_ids + target_ids[:remaining]
        return input_ids, len(context_ids)
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
        vocab_entropy: Optional[float] = None
        if send_to_user:
            if has_backbone and hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(hidden_out)
            else:
                logits = self.model(tokens)
            mx.eval(logits)
            last_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]
            token_id = int(mx.argmax(last_logits).item())
            synthesis_text = self.tokenizer.decode([token_id])
            # Real uncertainty: entropy of the actual next-token distribution.
            # Free here because the vocab projection is already paid for.
            p = mx.softmax(last_logits.astype(mx.float32))
            ent = -mx.sum(p * mx.log(p + 1e-10))
            mx.eval(ent)
            vocab_entropy = float(ent.item())
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
        entropy = vocab_entropy if vocab_entropy is not None else self.compute_reconstruction_entropy(synthesis_hidden)
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
        older callers working). One mx.eval + one host pull for the whole score.

        IMPORTANT — this is a heuristic, not ground truth: it only ever measures
        whether the expert agrees with Central's OWN hidden state, so it cannot
        tell a genuinely correct expert from one that confidently agrees with a
        hallucinated synthesis. It's the only option at live inference (a real
        user query has no known-correct continuation to check against). Whenever
        real target tokens ARE available (training/eval on real corpus text), use
        compute_grounded_r_i instead — that one is anchored to something outside
        the model."""
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
    def compute_r_i_batch(
        self,
        expert_outputs: List[Dict[str, Any]],
        contribution_hidden: mx.array,
        synthesis_hidden: mx.array,
        disagreement: Optional[Dict[int, float]] = None,
        loss_deltas: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """R_i as a WEIGHTED SUM, computed POST-SYNTHESIS over the whole batch.

        Four components per expert, each in [0, 1]:

          alignment   how much this expert's output moved Central toward what it
                      synthesised — direction + compatibility cosine
          no_halluc   1 - distance from the batch consensus; an expert alone in
                      left field is the confabulation signature
          speed       fastest expert's time / this expert's time
          perfection  grounded correctness from the leave-one-out loss delta,
                      present only when a real continuation exists

        WEIGHTS ARE DERIVED, not declared: each component is weighted by its
        spread across the experts in this batch. A component on which every
        expert scored the same separates nobody and earns no say; the component
        that actually spreads them apart carries the ranking. Same rule as
        composite TKL, so the two agree on what counts as informative.

        Batch-level by necessity — consensus and relative speed only exist across
        the set. compute_r_i() remains for the single-expert path, where only the
        alignment term is measurable.

        Feeds TKL as ONE component among seven, so it influences ranking without
        dominating it."""
        if not expert_outputs:
            return {}
        dis = disagreement or {}
        deltas = loss_deltas or {}
        times = [float(eo.get("wall_time", 0.0) or 0.0) for eo in expert_outputs]
        fastest = min([t for t in times if t > 0.0], default=0.0)
        parts: Dict[int, Dict[str, float]] = {}
        for eo in expert_outputs:
            eid = int(eo.get("expert_id", -1))
            comp: Dict[str, float] = {}
            align = self.compute_r_i(eo.get("hidden_states"), contribution_hidden,
                                     float(eo.get("wall_time", 0.0) or 0.0),
                                     synthesis_hidden=synthesis_hidden)
            comp["alignment"] = float(align)
            if eid in dis and dis[eid] is not None:
                comp["no_halluc"] = float(max(0.0, min(1.0, 1.0 - dis[eid])))
            t = float(eo.get("wall_time", 0.0) or 0.0)
            if fastest > 0.0 and t > 0.0:
                comp["speed"] = float(max(0.0, min(1.0, fastest / t)))
            if eid in deltas and deltas[eid] is not None:
                comp["perfection"] = float(self.compute_grounded_r_i(0.0, -float(deltas[eid])))
            parts[eid] = comp
        keys = ("alignment", "no_halluc", "speed", "perfection")
        spread = {}
        for k in keys:
            vals = [c[k] for c in parts.values() if k in c]
            spread[k] = float(np.std(vals)) if len(vals) > 1 else 0.0
        total = sum(spread.values())
        out: Dict[int, float] = {}
        for eid, comp in parts.items():
            have = [k for k in keys if k in comp]
            if not have:
                out[eid] = 0.0
            elif total > 1e-9 and sum(spread[k] for k in have) > 1e-9:
                w = {k: spread[k] for k in have}
                s = sum(w.values())
                out[eid] = float(max(0.0, min(1.0, sum(w[k] * comp[k] for k in have) / s)))
            else:
                out[eid] = float(np.mean([comp[k] for k in have]))
        return out
    def compute_grounded_r_i(self, loss_without_expert: float, loss_with_expert: float) -> float:
        """The REAL contribution score: did adding this expert's generated context
        to Central's prompt reduce the actual cross-entropy loss on the true
        continuation (training.compute_central_ce_loss, run once with and once
        without this expert's output_text in the context)? This is the fix for
        the closed self-agreement loop that compute_r_i's cosine terms can't
        escape — 'the expert agrees with Central' and 'the expert made Central
        more correct' are different claims, and this is the only one of the two
        that's checked against real text instead of the model's own output.

        delta = loss_without - loss_with; positive means the expert genuinely
        helped predict the real continuation, negative means it hurt. Squashed
        through a sigmoid to [0, 1] so it composes with compute_tkl the same way
        the cosine r_i did. Only computable when real target tokens exist
        (training/eval on labeled or real-corpus data) — there's no such thing as
        'the correct continuation' for a live, unanswered user query, so live
        inference still falls back to compute_r_i."""
        delta = loss_without_expert - loss_with_expert
        return float(1.0 / (1.0 + math.exp(-4.0 * delta)))
    def compute_l_eff(
        self,
        expert_output_hidden: mx.array,
        synthesis_hidden: mx.array,
        token_count: int,
        wall_time: float,
        loss_delta: Optional[float] = None,
    ) -> float:
        """Raw efficiency score for one expert (the gate's routing head learns to
        prefer high values). Per the manual: synthesis_compatibility + throughput.
        Returned un-normalised; the caller L1-normalises across the active experts
        before building the gate's L_eff target. Higher = more useful per second.

        Prefers `loss_delta` (loss_without_expert - loss_with_expert, from real
        target tokens) when the caller has it — that's genuine, ground-truth
        usefulness, not self-agreement. Falls back to the cosine-compatibility
        heuristic only when no real target exists (e.g. live inference)."""
        if loss_delta is not None:
            compatibility = max(0.0, loss_delta)
        else:
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
        """FALLBACK ONLY. Softmax entropy over hidden-state DIMENSIONS, which are
        not a distribution over anything — this number has no established relation
        to uncertainty. forward() now prefers the entropy of the real next-token
        distribution whenever the vocab projection has been computed; this path
        survives only for the measurement pass, where no logits exist."""
        if synthesis_hidden is None:
            return 0.0
        # Cast inside MLX before crossing to numpy. Qwen3 emits bfloat16 hidden
        # states, which numpy cannot view through the buffer protocol —
        # `np.asarray(bf16_array)` raises "Item size 2 ... does not match the
        # dtype B item size 1" and takes down every central.forward() call, i.e.
        # every Timeline B request. The old 4-bit Mistral happened to hand back a
        # numpy-viewable dtype, so the swap exposed this rather than caused it.
        values = np.asarray(synthesis_hidden.astype(mx.float32), dtype=np.float64).reshape(-1)
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
        #
        # The chat wrapper comes from the TOKENISER'S OWN template, never a
        # hardcoded one. This used to emit Mistral's "<s> [INST] ... [/INST]",
        # which is correct only while CENTRAL_MODEL_ID is a Mistral — swapping in
        # any other family (Qwen/ChatML, Llama, Gemma) silently degrades every
        # answer, because an instruct model handed a foreign template stops
        # behaving like an instruct model. Deriving it from the tokenizer makes
        # Central swappable, which the architecture depends on.
        user_content = input_text
        if expert_context:
            notes = "\n".join(f"- {c.strip()}" for c in expert_context if c and c.strip())
            if notes:
                user_content = (
                    f"{input_text}\n\n"
                    f"Expert analyses to consider:\n{notes}\n\n"
                    f"Use the analyses where they help and give the best final answer."
                )
        self.load()
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_template is not None and getattr(self.tokenizer, "chat_template", None):
            return apply_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_content   # no template available: send the raw question
    def generate(self, input_text: str, max_tokens: Optional[int] = None,
                 expert_context: Optional[List[str]] = None) -> str:
        self.load()
        from mlx_lm import generate as mlx_generate
        mt = max_tokens if max_tokens is not None else self.gen_max_tokens
        prompt = self.format_prompt(input_text, expert_context)
        return mlx_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=mt)

