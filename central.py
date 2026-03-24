# pyre-unsafe
"""Central model layer (Mistral-7B)."""
from __future__ import annotations

import aiohttp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import configs
import inference
from vectors import VectorBackend


@dataclass
class CentralOutput:
    output_text: str
    vector: np.ndarray


_vector_backend = VectorBackend(mode=configs.VECTOR_BACKEND, model_id=configs.VECTOR_MODEL_ID)


async def central_forward(text: str) -> CentralOutput:
    async with aiohttp.ClientSession() as session:
        if configs.USE_MOCK_INFERENCE:
            output_text = text
        else:
            try:
                output_text = await inference.hf_generate_async(
                    session=session,
                    model_id=configs.CENTRAL_MODEL_ID,
                    prompt=text,
                    max_new_tokens=128,
                    temperature=0.2,
                    do_sample=False,
                )
            except Exception as exc:
                if configs.DEBUG:
                    print(f"[central] HF call failed: {exc}")
                output_text = text
        vec_result = await _vector_backend.embed(
            session=session,
            text=output_text,
            target_dim=configs.CENTRAL_D_MODEL,
            salt="central",
        )
        return CentralOutput(output_text=output_text, vector=vec_result.vector)


def _align_vector(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.shape[0] == target_dim:
        return vec
    if vec.shape[0] > target_dim:
        return vec[:target_dim]
    pad = np.zeros(target_dim - vec.shape[0], dtype=vec.dtype)
    return np.concatenate([vec, pad], axis=0)


def attention_synthesis(
    expert_vectors: List[np.ndarray],
    token_embedding: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    if not expert_vectors:
        return token_embedding
    d = token_embedding.shape[0]
    aligned = [_align_vector(v, d) for v in expert_vectors]
    keys = np.stack(aligned, axis=0)
    values = keys.copy()
    query = token_embedding

    scale = np.sqrt(float(d)) * temperature
    scores = keys @ query / (scale + 1e-9)

    scores = scores - scores.max()
    weights = np.exp(scores)
    weights = weights / (weights.sum() + 1e-9)

    synthesized = (weights[:, None] * values).sum(axis=0)

    alpha = 0.7
    output = alpha * synthesized + (1.0 - alpha) * token_embedding
    output = output / (np.linalg.norm(output) + 1e-9) * np.linalg.norm(token_embedding)
    return output


def gate_optimization_step(
    expert_vectors: List[np.ndarray],
    central_vector: Union[np.ndarray, None] = None,
) -> Dict[str, Any]:
    if not expert_vectors:
        return {"updated": False, "quality": 0.0, "diversity": 0.0, "signals": []}

    norms = [np.linalg.norm(v) for v in expert_vectors]
    mean_norm = float(np.mean(norms))

    quality_scores = []
    if central_vector is not None:
        c_norm = np.linalg.norm(central_vector)
        for v in expert_vectors:
            v_norm = np.linalg.norm(v)
            if c_norm > 1e-9 and v_norm > 1e-9:
                d_min = min(v.shape[0], central_vector.shape[0])
                cos_sim = float(np.dot(v[:d_min], central_vector[:d_min]) / (v_norm * c_norm))
                quality_scores.append(cos_sim)
            else:
                quality_scores.append(0.0)
    else:
        quality_scores = [1.0] * len(expert_vectors)

    pairwise_dists = []
    for i in range(len(expert_vectors)):
        for j in range(i + 1, len(expert_vectors)):
            vi, vj = expert_vectors[i], expert_vectors[j]
            ni, nj = norms[i], norms[j]
            if ni > 1e-9 and nj > 1e-9:
                d_min = min(vi.shape[0], vj.shape[0])
                cos = float(np.dot(vi[:d_min], vj[:d_min]) / (ni * nj))
                pairwise_dists.append(1.0 - cos)
    diversity = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0

    return {
        "updated": True,
        "quality": float(np.mean(quality_scores)),
        "diversity": diversity,
        "mean_norm": mean_norm,
        "signals": quality_scores,
    }


def teacher_distillation_signal(
    central_output: np.ndarray,
    temperature: float = 2.0,
) -> Dict[str, Any]:
    logits = central_output / (temperature + 1e-9)
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    soft_labels = exp_logits / (exp_logits.sum() + 1e-9)

    entropy = float(-np.sum(soft_labels * np.log(soft_labels + 1e-12)))
    top_k_indices = np.argsort(soft_labels)[-5:][::-1].tolist()

    return {
        "soft_labels": soft_labels,
        "signal_norm": float(np.linalg.norm(central_output)),
        "entropy": entropy,
        "top_k_indices": top_k_indices,
        "temperature": temperature,
    }


def self_test() -> None:
    print("[central] self-test")
    import asyncio

    async def _run():
        out = await central_forward("hello central")
        print(f"[central] output_text_len={len(out.output_text)}")

        expert_vecs = [
            np.random.randn(configs.CENTRAL_D_MODEL).astype(np.float32)
            for _ in range(4)
        ]
        synth = attention_synthesis(expert_vecs, out.vector)
        print(f"[central] synthesis_dim={synth.shape[0]}")

        gate_update = gate_optimization_step(expert_vecs, central_vector=synth)
        print(f"[central] gate_quality={gate_update['quality']:.3f} diversity={gate_update['diversity']:.3f}")

        distill = teacher_distillation_signal(synth)
        print(f"[central] distill_entropy={distill['entropy']:.3f} signal_norm={distill['signal_norm']:.3f}")

    asyncio.run(_run())


if __name__ == "__main__":
    self_test()
