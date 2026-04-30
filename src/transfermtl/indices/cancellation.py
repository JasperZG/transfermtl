"""Cancellation index C_ij (plan §2.9).

C_ij = Σ_r w_r · |Δ(r)|  /  (|Σ_r w_r · Δ(r)| + η)

with weights `w_r` proportional to the regional test-set size. The η term
(default 0.5) prevents division by zero when regional Δ values exactly
cancel. C ≈ 1 when Δ has uniform sign; C grows large when sign flips across
regions. C > 1 indicates regional magnitudes substantially exceed the global
average.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def compute_C(
    deltas: Mapping[int, float],
    region_test_sizes: Mapping[int, int],
    eta: float = 0.5,
) -> float:
    """Numerator (sum of weighted |Δ|) over (|sum of weighted Δ| + η)."""
    region_ids = sorted(deltas.keys())
    if not region_ids:
        return float("nan")
    d = np.array([float(deltas[r]) for r in region_ids], dtype=np.float64)
    w = np.array([float(region_test_sizes.get(r, 0)) for r in region_ids], dtype=np.float64)
    valid = ~np.isnan(d) & (w > 0)
    if not np.any(valid):
        return float("nan")
    d, w = d[valid], w[valid]

    numerator = float(np.sum(w * np.abs(d)))
    signed = float(np.sum(w * d))
    denominator = float(abs(signed) + eta)
    if denominator <= 0:
        return float("nan")
    return numerator / denominator
