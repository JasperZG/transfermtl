"""Empirical p-value with smoothing (plan §2.13)."""

from __future__ import annotations

import numpy as np


def empirical_pvalue(observed: float, null: np.ndarray) -> float:
    """Plug-in p-value: (1 + #{b : null[b] >= observed}) / (1 + B).

    The +1 in numerator/denominator is the standard "add-one smoothing" used
    in permutation tests; it ensures p > 0 even when no null sample exceeds
    the observed value, and keeps the test well-defined for B = 0
    (degenerates to p = 1).
    """
    arr = np.asarray(null, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    b = arr.size
    if b == 0:
        return 1.0
    return float((1 + int(np.sum(arr >= observed))) / (1 + b))
