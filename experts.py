# pyre-unsafe
"""Expert pool management."""
from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import aiohttp
import numpy as np

import configs
import inference
from vectors import VectorBackend


@dataclass
class ExpertOutput:
    index: int
    output_text: str
    vector: np.ndarray
    hidden_states: np.ndarray
    from_cache: bool


class LRUCache:
    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size
        self._data: OrderedDict[str, ExpertOutput] = OrderedDict()

    def get(self, key: str) -> Optional[ExpertOutput]:
        value = self._data.get(key)
        if value is None:
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: ExpertOutput) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)


class ExpertPool:
    def __init__(self, max_cache: int = 2048) -> None:
        self.cache = LRUCache(max_size=max_cache)
        self.utilization_counts = [0 for _ in range(configs.NUM_EXPERTS)]
        self.utilization_window: deque[List[int]] = deque(maxlen=configs.UTILIZATION_WINDOW)
        self.vector_backend = VectorBackend(mode=configs.VECTOR_BACKEND, model_id=configs.VECTOR_MODEL_ID)

    def _hash_input(self, text: str, context: str, expert_idx: int) -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(context.encode("utf-8"))
        h.update(str(expert_idx).encode("utf-8"))
        return h.hexdigest()

    async def _call_hf(self, model_id: str, text: str, session: aiohttp.ClientSession) -> str:
        return await inference.hf_generate_async(
            session=session,
            model_id=model_id,
            prompt=text,
            max_new_tokens=64,
            temperature=0.2,
            do_sample=False,
        )

    async def expert_forward(self, expert_idx: int, text: str, context: str, session: aiohttp.ClientSession) -> ExpertOutput:
        cache_key = self._hash_input(text, context, expert_idx)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return ExpertOutput(
                index=cached.index,
                output_text=cached.output_text,
                vector=cached.vector,
                hidden_states=cached.hidden_states,
                from_cache=True,
            )

        if configs.USE_MOCK_INFERENCE:
            output_text = text
        else:
            try:
                model_ref = f"expert_{expert_idx:03d}" if configs.NATIVE_MODE else configs.EXPERT_MODEL_ID
                output_text = await self._call_hf(model_ref, text, session)
            except Exception as exc:
                if configs.DEBUG:
                    print(f"[experts] HF call failed for expert {expert_idx}: {exc}")
                output_text = text

        vec_result = await self.vector_backend.embed(
            session=session,
            text=output_text,
            target_dim=configs.EXPERT_D_MODEL,
            salt=f"expert-{expert_idx}",
        )
        vector = vec_result.vector
        hidden_states = vector.copy()
        output = ExpertOutput(
            index=expert_idx,
            output_text=output_text,
            vector=vector,
            hidden_states=hidden_states,
            from_cache=False,
        )
        self.cache.set(cache_key, output)
        return output

    async def run_experts(
        self,
        expert_indices: List[int],
        text: str,
        context: str,
        x_concurrency: int = configs.X_DEFAULT,
    ) -> List[ExpertOutput]:
        if not expert_indices:
            return []

        outputs: List[ExpertOutput] = []
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(expert_indices), x_concurrency):
                batch = expert_indices[i : i + x_concurrency]
                tasks = [self.expert_forward(idx, text, context, session) for idx in batch]
                batch_out = await asyncio.gather(*tasks)
                outputs.extend(batch_out)
        self.update_utilization([o.index for o in outputs])
        return outputs

    def update_utilization(self, expert_indices: List[int]) -> None:
        if len(self.utilization_window) == self.utilization_window.maxlen:
            old = self.utilization_window.popleft()
            for idx in old:
                self.utilization_counts[idx] -= 1
        self.utilization_window.append(expert_indices)
        for idx in expert_indices:
            self.utilization_counts[idx] += 1

    def utilization_rates(self) -> List[float]:
        total = sum(self.utilization_counts) or 1
        return [count / total for count in self.utilization_counts]

    def least_used(self, threshold: float = configs.UNDERUSE_THRESHOLD) -> List[int]:
        rates = self.utilization_rates()
        return [i for i, r in enumerate(rates) if r < threshold]

    def expert_communication_phase(
        self,
        expert_outputs: List[ExpertOutput],
        central_output: np.ndarray,
        input_text: str,
        context: str,
    ) -> Dict[str, Any]:
        if not expert_outputs:
            return {"updated": False, "participants": [], "diversity": 0.0}

        k = len(expert_outputs)
        vectors = [o.vector for o in expert_outputs]
        norms = [np.linalg.norm(v) for v in vectors]

        c_norm = np.linalg.norm(central_output)
        quality_scores = []
        for i, v in enumerate(vectors):
            if norms[i] > 1e-9 and c_norm > 1e-9:
                d_min = min(v.shape[0], central_output.shape[0])
                cos = float(np.dot(v[:d_min], central_output[:d_min]) / (norms[i] * c_norm))
                quality_scores.append(cos)
            else:
                quality_scores.append(0.0)

        peer_penalties = [0.0] * k
        pairwise_sims: List[float] = []
        for i in range(k):
            for j in range(i + 1, k):
                vi, vj = vectors[i], vectors[j]
                d_min = min(vi.shape[0], vj.shape[0])
                if norms[i] > 1e-9 and norms[j] > 1e-9:
                    sim = float(np.dot(vi[:d_min], vj[:d_min]) / (norms[i] * norms[j]))
                else:
                    sim = 0.0
                pairwise_sims.append(sim)
                peer_penalties[i] += sim
                peer_penalties[j] += sim

        if k > 1:
            peer_penalties = [p / (k - 1) for p in peer_penalties]

        combined_signals = []
        for i in range(k):
            signal = quality_scores[i] - 0.5 * peer_penalties[i]
            combined_signals.append(signal)

        diversity = 1.0 - (float(np.mean(pairwise_sims)) if pairwise_sims else 0.0)
        mean_quality = float(np.mean(quality_scores))
        unique_count = len({o.index for o in expert_outputs})

        return {
            "updated": True,
            "participants": [o.index for o in expert_outputs],
            "diversity": diversity,
            "mean_quality": mean_quality,
            "quality_scores": quality_scores,
            "peer_penalties": peer_penalties,
            "combined_signals": combined_signals,
            "unique_experts": unique_count,
        }

    def expert_feedback_loop(
        self,
        expert_outputs: List[ExpertOutput],
        central_vector: np.ndarray,
        max_rounds: int = 3,
        convergence_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        if len(expert_outputs) < 2:
            return {
                "rounds": 0,
                "converged": True,
                "final_vectors": [o.vector for o in expert_outputs],
                "quality_trajectory": [],
            }

        vectors = [o.vector.copy().astype(np.float64) for o in expert_outputs]
        k = len(vectors)
        quality_trajectory: List[float] = []

        for round_num in range(max_rounds):
            aggregate = np.mean(vectors, axis=0)

            c_norm = np.linalg.norm(central_vector)
            quality_scores = []
            for v in vectors:
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-9 and c_norm > 1e-9:
                    d_min = min(v.shape[0], central_vector.shape[0])
                    cos = float(np.dot(v[:d_min], central_vector[:d_min]) / (v_norm * c_norm))
                    quality_scores.append(cos)
                else:
                    quality_scores.append(0.0)
            mean_quality = float(np.mean(quality_scores))
            quality_trajectory.append(mean_quality)

            new_vectors = []
            for i, v in enumerate(vectors):
                peer_mean = np.mean([vectors[j] for j in range(k) if j != i], axis=0)
                alpha = 0.15
                beta = 0.10
                refined = v + alpha * (peer_mean - v) + beta * (central_vector[:v.shape[0]] - v)
                norm = np.linalg.norm(refined)
                if norm > 1e-9:
                    refined = refined / norm * np.linalg.norm(v)
                new_vectors.append(refined)

            max_delta = max(
                float(np.linalg.norm(new_vectors[i] - vectors[i]))
                for i in range(k)
            )
            vectors = new_vectors

            if max_delta < convergence_threshold:
                break

        for i, out in enumerate(expert_outputs):
            out.vector = vectors[i].astype(np.float32)

        return {
            "rounds": round_num + 1,
            "converged": max_delta < convergence_threshold,
            "final_vectors": vectors,
            "quality_trajectory": quality_trajectory,
        }


def self_test() -> None:
    print("[experts] self-test")
    pool = ExpertPool(max_cache=32)
    text = "hello"
    context = "test"

    async def _run():
        outs = await pool.run_experts([0, 1, 2, 3], text, context, x_concurrency=2)
        print(f"[experts] outputs: {len(outs)}")
        print(f"[experts] cache hit: {pool.cache.get(pool._hash_input(text, context, 0)) is not None}")

        central_vec = np.random.randn(configs.CENTRAL_D_MODEL).astype(np.float32)
        comm = pool.expert_communication_phase(outs, central_vec, text, context)
        print(f"[experts] diversity={comm['diversity']:.3f} quality={comm['mean_quality']:.3f}")

        feedback = pool.expert_feedback_loop(outs, central_vec, max_rounds=3)
        print(f"[experts] feedback rounds={feedback['rounds']} converged={feedback['converged']}")
        print(f"[experts] quality trajectory={[f'{q:.3f}' for q in feedback['quality_trajectory']]}")

    asyncio.run(_run())


if __name__ == "__main__":
    self_test()
