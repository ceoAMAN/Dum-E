# pyre-unsafe
"""Vector backend for expert/central representations."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import aiohttp
import numpy as np

import configs
import inference


@dataclass
class VectorResult:
    vector: np.ndarray
    backend: str
    ok: bool


def _hash_vector(text: str, dim: int, salt: str) -> np.ndarray:
    seed = int(hashlib.sha256((text + salt).encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _align(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.shape[0] == target_dim:
        return vec
    if vec.shape[0] > target_dim:
        return vec[:target_dim]
    pad = np.zeros(target_dim - vec.shape[0], dtype=vec.dtype)
    return np.concatenate([vec, pad], axis=0)


def _pool_feature_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=0)
    if arr.ndim == 3:
        return arr.mean(axis=0).mean(axis=0)
    return arr.reshape(-1)


class VectorBackend:
    def __init__(self, mode: str = "hash", model_id: Optional[str] = None) -> None:
        self.mode = mode
        self.model_id = model_id or configs.CENTRAL_MODEL_ID

    async def embed(
        self,
        session: aiohttp.ClientSession,
        text: str,
        target_dim: int,
        salt: str,
    ) -> VectorResult:
        if self.mode == "hash":
            vec = _hash_vector(text, target_dim, salt)
            return VectorResult(vector=vec, backend="hash", ok=True)

        if self.mode in {"hf", "hf_feature"}:
            raw = await inference.hf_feature_extract_async(session, self.model_id, text)
            if raw is None:
                vec = _hash_vector(text, target_dim, salt)
                return VectorResult(vector=vec, backend="hash", ok=False)
            try:
                arr = np.array(raw, dtype=np.float32)
            except Exception:
                vec = _hash_vector(text, target_dim, salt)
                return VectorResult(vector=vec, backend="hash", ok=False)
            pooled = _pool_feature_array(arr)
            vec = _align(pooled.astype(np.float32), target_dim)
            return VectorResult(vector=vec, backend="hf_feature", ok=True)

        vec = _hash_vector(text, target_dim, salt)
        return VectorResult(vector=vec, backend="hash", ok=False)
