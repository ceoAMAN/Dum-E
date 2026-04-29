from __future__ import annotations
import numpy as np
import mlx.core as mx
def mx_to_numpy(arr: mx.array) -> np.ndarray:
    return np.array(arr.tolist(), dtype=np.float32)
def cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
    a_norm = a / (mx.linalg.norm(a) + 1e-8)
    b_norm = b / (mx.linalg.norm(b) + 1e-8)
    return mx.sum(a_norm * b_norm)
def cosine_distance(a: mx.array, b: mx.array) -> mx.array:
    return 1.0 - cosine_similarity(a, b)
def numpy_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a) + 1e-8
    b_norm = np.linalg.norm(b) + 1e-8
    return float(1.0 - np.dot(a, b) / (a_norm * b_norm))
def batch_cosine_distances(query: mx.array, matrix: mx.array) -> mx.array:
    q_norm = query / (mx.linalg.norm(query) + 1e-8)
    m_norm = matrix / (mx.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    similarities = mx.matmul(m_norm, q_norm)
    return 1.0 - similarities
def dot_product_similarity_matrix(weight_matrices: list[mx.array]) -> mx.array:
    flat = mx.stack([w.reshape(-1) for w in weight_matrices], axis=0)
    norms = mx.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    normed = flat / norms
    return mx.matmul(normed, normed.T)
def normalise_scores(scores: mx.array, eps: float = 1e-8) -> mx.array:
    total = mx.sum(mx.abs(scores)) + eps
    return scores / total
def exponential_decay_weights(n: int, gamma: float = 0.95) -> mx.array:
    exponents = mx.array(list(range(n - 1, -1, -1)), dtype=mx.float32)
    return mx.power(mx.array(gamma, dtype=mx.float32), exponents)
def compute_mean_inter_centroid_distance(centroids: list[np.ndarray]) -> float:
    n = len(centroids)
    if n < 2:
        return 1.0
    matrix = np.stack(centroids, axis=0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    normed = matrix / norms
    sim_matrix = normed @ normed.T
    dist_matrix = 1.0 - sim_matrix
    upper = dist_matrix[np.triu_indices(n, k=1)]
    return float(upper.mean()) if len(upper) > 0 else 1.0
