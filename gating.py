# pyre-unsafe
"""Routing and timeline execution."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import configs
import central
from experts import ExpertPool
import inference
from memory import RoutingMemory


@dataclass
class GateOutput:
    logits: np.ndarray
    confidence: float


@dataclass
class RoutingDecision:
    k: int
    expert_indices: List[int]
    timeline: str
    confidence: float
    x: int
    y: int


@dataclass
class GateLLMDecision:
    k: int
    expert_indices: List[int]
    confidence: float


class RoutingEMA:
    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.fast_path_rate = 0.0
        self.entropy = 0.0
        self.per_expert_freq = np.zeros(configs.NUM_EXPERTS, dtype=np.float64)

    def update(self, fast_path: bool, entropy: float, expert_indices: Optional[List[int]] = None) -> None:
        self.fast_path_rate = self.decay * self.fast_path_rate + (1 - self.decay) * float(fast_path)
        self.entropy = self.decay * self.entropy + (1 - self.decay) * entropy
        if expert_indices:
            indicator = np.zeros(configs.NUM_EXPERTS, dtype=np.float64)
            for idx in expert_indices:
                if 0 <= idx < configs.NUM_EXPERTS:
                    indicator[idx] = 1.0
            self.per_expert_freq = self.decay * self.per_expert_freq + (1 - self.decay) * indicator


class GateRouter:
    def __init__(self, expert_pool: ExpertPool) -> None:
        self.expert_pool = expert_pool
        self.fast_path_threshold = configs.FAST_PATH_THRESHOLD
        self.ema = RoutingEMA(decay=configs.EMA_DECAY)
        self.expert_bias = np.zeros(configs.NUM_EXPERTS, dtype=np.float32)
        self.memory = RoutingMemory()
        mem_path = configs.CHECKPOINT_DIR / "routing_memory.json"
        self.memory.load(mem_path)

    def _mock_logits(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(configs.NUM_EXPERTS).astype(np.float32)

    def forward(self, text: str, context: str) -> GateOutput:
        if not configs.USE_MOCK_INFERENCE and configs.GATE_MODE == "llm":
            llm_decision = self._call_gate_llm(text, context)
            if llm_decision is not None:
                logits = np.full(configs.NUM_EXPERTS, -3.0, dtype=np.float32)
                for idx in llm_decision.expert_indices:
                    if 0 <= idx < configs.NUM_EXPERTS:
                        logits[idx] = 2.0
                return GateOutput(logits=logits, confidence=llm_decision.confidence)

        logits = self._mock_logits(text + "|" + context)
        probs = _softmax(logits)
        confidence = float(probs.max())
        return GateOutput(logits=logits, confidence=confidence)

    def _call_gate_llm(self, text: str, context: str) -> Optional[GateLLMDecision]:
        if not configs.HF_TOKEN:
            return None
        prompt = (
            "You are the Sturnus Gate. Return ONLY JSON with keys: "
            "\"confidence\" (0-1), \"k\" (0-20), \"expert_indices\" "
            "(list of ints length k, each 0-49). No other text.\\n"
            f"Input: {text}\\nContext: {context}"
        )
        generated = inference.hf_generate_sync(
            model_id=configs.GATE_MODEL_ID,
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
        )
        if not generated:
            if configs.DEBUG:
                print("[gating] HF gate returned empty response")
            return None
        parsed = _parse_gate_json(generated)
        if parsed is None:
            return None
        k = int(parsed.get("k", len(parsed.get("expert_indices", []))))
        confidence = float(parsed.get("confidence", 0.0))
        expert_indices = parsed.get("expert_indices", [])
        return GateLLMDecision(k=k, expert_indices=expert_indices, confidence=confidence)

    def _signals(self, text: str, context: str, probs: np.ndarray) -> Tuple[float, float, float, float]:
        token_len = len(text.split())
        variance = min(1.0, token_len / 8.0)
        complexity = min(1.0, (len(text) + len(context)) / 400.0)
        context_signal = min(1.0, len(context) / 2000.0)
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)) / np.log(len(probs)))
        return variance, complexity, context_signal, entropy

    def dynamic_top_k(self, variance: float, complexity: float, context_signal: float, entropy: float) -> int:
        score = (
            0.30 * variance
            + 0.35 * complexity
            + 0.25 * entropy
            + 0.10 * (1.0 - context_signal)
        )
        k = int(round(configs.K_MIN + score * (configs.K_MAX - configs.K_MIN)))
        return int(np.clip(k, configs.K_MIN, configs.K_MAX))

    def select_experts(self, logits: np.ndarray, k: int, confidence: float, prompt: str = "") -> List[int]:
        if k <= 0:
            return []
        reward_scores = self.memory.reward_tracker.get_expert_scores()
        reward_boost = (reward_scores / (reward_scores.max() + 1e-9)).astype(np.float32)
        adjusted = logits + self.expert_bias + 0.3 * reward_boost

        if prompt:
            memory_suggestion = self.memory.suggest_experts(prompt, k)
            if memory_suggestion:
                for idx in memory_suggestion[:k // 2]:
                    if 0 <= idx < configs.NUM_EXPERTS:
                        adjusted[idx] += 1.0

        top_indices = list(np.argsort(adjusted)[-k:][::-1])

        mask_slots = max(1, int(round(k * configs.MASKING_RATE))) if k > 0 else 0
        if mask_slots > 0:
            available = [i for i in range(configs.NUM_EXPERTS) if i not in top_indices]
            rng = np.random.default_rng(int(confidence * 1000) + k)
            rng.shuffle(available)
            for i in range(min(mask_slots, len(top_indices), len(available))):
                top_indices[-(i + 1)] = available[i]

        underused = self.expert_pool.least_used()
        if underused:
            if confidence < 0.30:
                replace = max(1, k // 2)
                for i in range(replace):
                    top_indices[i] = underused[i % len(underused)]
            elif confidence < 0.55:
                top_indices[-1] = underused[0]

        seen: set = set()
        deduped = []
        for idx in top_indices:
            if idx not in seen:
                seen.add(idx)
                deduped.append(int(idx))
        return deduped[:k]

    def _apply_anti_collapse(self, expert_indices: List[int], k: int, confidence: float) -> List[int]:
        if k <= 0:
            return []
        indices = [int(i) for i in expert_indices if 0 <= int(i) < configs.NUM_EXPERTS]
        if len(indices) < k:
            remaining = [i for i in range(configs.NUM_EXPERTS) if i not in indices]
            remaining_sorted = sorted(remaining, key=lambda i: float(self.expert_bias[i]), reverse=True)
            indices.extend(remaining_sorted[: k - len(indices)])

        mask_slots = max(1, int(round(k * configs.MASKING_RATE))) if k > 0 else 0
        if mask_slots > 0:
            available = [i for i in range(configs.NUM_EXPERTS) if i not in indices]
            rng = np.random.default_rng(int(confidence * 1000) + k)
            rng.shuffle(available)
            for i in range(min(mask_slots, len(indices), len(available))):
                indices[-(i + 1)] = int(available[i])

        underused = self.expert_pool.least_used()
        if underused:
            if confidence < 0.30:
                replace = max(1, k // 2)
                for i in range(replace):
                    indices[i] = underused[i % len(underused)]
            elif confidence < 0.55:
                indices[-1] = underused[0]

        seen: set = set()
        deduped = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                deduped.append(int(idx))
        return deduped[:k]

    def route(self, text: str, context: str) -> RoutingDecision:
        if not configs.USE_MOCK_INFERENCE and configs.GATE_MODE == "llm":
            llm_decision = self._call_gate_llm(text, context)
            if llm_decision is not None:
                k = int(np.clip(llm_decision.k, configs.K_MIN, configs.K_MAX))
                confidence = float(np.clip(llm_decision.confidence, 0.0, 1.0))
                expert_indices = self._apply_anti_collapse(llm_decision.expert_indices, k, confidence)
                x, y = _determine_xy(k, confidence)
                timeline = "A" if confidence >= self.fast_path_threshold or k == 0 else "B"
                entropy = min(1.0, k / max(configs.K_MAX, 1))
                self.ema.update(fast_path=(timeline == "A"), entropy=entropy, expert_indices=expert_indices)
                self._self_stabilize()
                return RoutingDecision(k=k, expert_indices=expert_indices, timeline=timeline, confidence=confidence, x=x, y=y)

        gate_out = self.forward(text, context)
        probs = _softmax(gate_out.logits)
        variance, complexity, context_signal, entropy = self._signals(text, context, probs)
        k = self.dynamic_top_k(variance, complexity, context_signal, entropy)
        expert_indices = self.select_experts(gate_out.logits, k, gate_out.confidence, prompt=text)
        x, y = _determine_xy(k, gate_out.confidence)
        timeline = "A" if gate_out.confidence >= self.fast_path_threshold or k == 0 else "B"

        self.ema.update(fast_path=(timeline == "A"), entropy=entropy, expert_indices=expert_indices)
        self._self_stabilize()

        return RoutingDecision(k=k, expert_indices=expert_indices, timeline=timeline, confidence=gate_out.confidence, x=x, y=y)

    def _self_stabilize(self) -> None:
        if self.ema.fast_path_rate > 0.80:
            self.fast_path_threshold = max(0.50, self.fast_path_threshold - 0.01)
        elif self.ema.fast_path_rate < 0.50:
            self.fast_path_threshold = min(0.95, self.fast_path_threshold + 0.01)

        mean_freq = float(self.ema.per_expert_freq.mean())
        if mean_freq > 1e-9:
            relative = self.ema.per_expert_freq / (mean_freq + 1e-9)
            adjustment = np.clip(1.0 - relative, -0.5, 0.5).astype(np.float32)
            self.expert_bias = np.clip(self.expert_bias + 0.01 * adjustment, -2.0, 2.0)

    async def run_timeline_a(self, text: str) -> Dict[str, Any]:
        central_out = await central.central_forward(text)
        return {
            "output_text": central_out.output_text,
            "central_vector": central_out.vector,
            "mode": "A",
        }

    async def run_timeline_b(
        self,
        text: str,
        context: str,
        expert_indices: List[int],
        mode: int = 2,
        x_concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        expert_outputs = await self.expert_pool.run_experts(
            expert_indices,
            text,
            context,
            x_concurrency=x_concurrency or configs.X_DEFAULT,
        )

        central_out = await central.central_forward(text)

        gate_update = central.gate_optimization_step(
            [o.vector for o in expert_outputs],
            central_vector=central_out.vector,
        )

        synth = central.attention_synthesis(
            [o.vector for o in expert_outputs],
            central_out.vector,
        )

        feedback = self.expert_pool.expert_feedback_loop(
            expert_outputs,
            synth,
            max_rounds=3,
        )

        distill = central.teacher_distillation_signal(synth)

        expert_update = self.expert_pool.expert_communication_phase(
            expert_outputs,
            synth,
            input_text=text,
            context=context,
        )

        quality = expert_update.get("mean_quality", 0.0)
        diversity = expert_update.get("diversity", 0.0)
        self.memory.store(
            prompt=text,
            expert_indices=expert_indices,
            quality=quality,
            diversity=diversity,
            k=len(expert_indices),
            timeline=f"B{mode}",
        )

        return {
            "output_text": central_out.output_text,
            "synth_vector": synth,
            "gate_update": gate_update,
            "distill": distill,
            "expert_update": expert_update,
            "feedback": feedback,
            "mode": f"B{mode}",
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / (exp.sum() + 1e-9)


def _parse_gate_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _determine_xy(k: int, confidence: float) -> Tuple[int, int]:
    if k <= 0:
        return 0, 0
    complexity = 1.0 - float(np.clip(confidence, 0.0, 1.0))
    x = int(round(configs.X_DEFAULT + complexity * (configs.X_MAX - configs.X_DEFAULT)))
    x = int(np.clip(x, 1, configs.X_MAX))
    y = int(math.ceil(k / x)) if x > 0 else 0
    if y > configs.Y_MAX:
        x = int(np.clip(math.ceil(k / configs.Y_MAX), 1, configs.X_MAX))
        y = int(math.ceil(k / x)) if x > 0 else 0
    return x, y


def self_test() -> None:
    print("[gating] self-test")
    pool = ExpertPool(max_cache=16)
    router = GateRouter(pool)
    decision = router.route("hello world", "context")
    print(f"[gating] K={decision.k} X={decision.x} Y={decision.y} timeline={decision.timeline} conf={decision.confidence:.3f}")

    async def _run():
        if decision.timeline == "A":
            result = await router.run_timeline_a("hello world")
        else:
            result = await router.run_timeline_b("hello world", "context", decision.expert_indices, mode=2)
        print(f"[gating] result keys: {list(result.keys())}")
        if "feedback" in result:
            fb = result["feedback"]
            print(f"[gating] feedback rounds={fb['rounds']} converged={fb['converged']}")

    asyncio.run(_run())

    stats = router.memory.get_stats()
    print(f"[gating] memory records={stats['total_records']} top_experts={stats['top_experts'][:5]}")


if __name__ == "__main__":
    self_test()
