"""Regional gradient affinity G_ij(r) = cos(g_i(r), g_j(r)).

NaN guard: the configs/_shared/preprocess.yaml `grad_norm_zero` threshold
(1e-8) decides when a gradient is degenerate. Below it cosine is undefined
and we return NaN rather than divide by zero.
"""

from __future__ import annotations

import numpy as np

GRAD_NORM_ZERO = 1.0e-8


def cosine_affinity(
    g_i: np.ndarray,
    g_j: np.ndarray,
    grad_norm_zero: float = GRAD_NORM_ZERO,
) -> float:
    """Cosine of two gradient vectors. Returns NaN if either norm < threshold."""
    g_i = np.asarray(g_i, dtype=np.float64).flatten()
    g_j = np.asarray(g_j, dtype=np.float64).flatten()
    if g_i.shape != g_j.shape:
        raise ValueError(f"shape mismatch: {g_i.shape} vs {g_j.shape}")
    n_i = float(np.linalg.norm(g_i))
    n_j = float(np.linalg.norm(g_j))
    if n_i < grad_norm_zero or n_j < grad_norm_zero:
        return float("nan")
    return float(np.dot(g_i, g_j) / (n_i * n_j))


def dot_product_affinity(g_i: np.ndarray, g_j: np.ndarray) -> float:
    """Unnormalized dot product. Cached for ablation experiments (plan §2.7)."""
    g_i = np.asarray(g_i, dtype=np.float64).flatten()
    g_j = np.asarray(g_j, dtype=np.float64).flatten()
    if g_i.shape != g_j.shape:
        raise ValueError(f"shape mismatch: {g_i.shape} vs {g_j.shape}")
    return float(np.dot(g_i, g_j))
