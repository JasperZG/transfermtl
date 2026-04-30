"""Stability of G_ij(r) across training (plan §2.7 step 4).

Three checkpoint snapshots are evaluated: `final`, `0.8`, `0.6` (interpreted as
80% / 60% of total epochs). The wrapper takes a callable that loads the model
at a given fraction so callers can control how trajectories are stored.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from transfermtl.gradients.affinity import cosine_affinity

CheckpointLabel = str  # "final" | "0.8" | "0.6"
DEFAULT_LABELS: tuple[CheckpointLabel, ...] = ("final", "0.8", "0.6")


def compute_trajectory_affinity(
    grad_loader: Callable[[CheckpointLabel], tuple[np.ndarray, np.ndarray]],
    labels: tuple[CheckpointLabel, ...] = DEFAULT_LABELS,
) -> dict[CheckpointLabel, dict[str, Any]]:
    """Compute G_ij(r) at three checkpoint snapshots.

    `grad_loader(label)` must return (g_i, g_j) for that snapshot. Returns
    `{label: {"G_ij": float, "g_i_norm": float, "g_j_norm": float}}` plus a
    summary entry under the key `"_summary"` with mean / std across labels.
    """
    out: dict[CheckpointLabel, dict[str, Any]] = {}
    g_values: list[float] = []
    for label in labels:
        g_i, g_j = grad_loader(label)
        G = cosine_affinity(g_i, g_j)
        out[label] = {
            "G_ij": G,
            "g_i_norm": float(np.linalg.norm(g_i)),
            "g_j_norm": float(np.linalg.norm(g_j)),
        }
        if not np.isnan(G):
            g_values.append(G)

    if g_values:
        out["_summary"] = {
            "mean": float(np.mean(g_values)),
            "std": float(np.std(g_values, ddof=0)),
            "n": len(g_values),
        }
    else:
        out["_summary"] = {"mean": float("nan"), "std": float("nan"), "n": 0}

    return out
