"""Percentile bootstrap CI (plan §2.10)."""

from __future__ import annotations

import numpy as np


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided percentile CI at level (1 - alpha).

    Returns (lo, hi) where lo = quantile(samples, alpha/2),
    hi = quantile(samples, 1 - alpha/2). NaNs are dropped before
    quantile computation. Empty input returns (nan, nan).
    """
    arr = np.asarray(samples, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return lo, hi
