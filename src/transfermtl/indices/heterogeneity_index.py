"""Heterogeneity index H_ij (plan §2.9).

H_ij = test-set-weighted mean absolute deviation of regional Δ from the
test-set-weighted global mean. Returns 0 when Δ is constant across regions
and grows with the spread of regional benefits.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def compute_H(
    deltas: Mapping[int, float],
    region_test_sizes: Mapping[int, int],
) -> float:
    """Test-set-weighted MAD of `deltas` from the weighted mean."""
    region_ids = sorted(deltas.keys())
    if not region_ids:
        return float("nan")

    d = np.array([float(deltas[r]) for r in region_ids], dtype=np.float64)
    w = np.array([float(region_test_sizes.get(r, 0)) for r in region_ids], dtype=np.float64)

    valid = ~np.isnan(d) & (w > 0)
    if not np.any(valid):
        return float("nan")
    d, w = d[valid], w[valid]

    total_w = float(w.sum())
    if total_w <= 0:
        return float("nan")
    mean = float(np.sum(d * w) / total_w)
    mad = float(np.sum(w * np.abs(d - mean)) / total_w)
    return mad
