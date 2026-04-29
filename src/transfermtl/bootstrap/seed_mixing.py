"""Seed-mixing utility for hierarchical bootstrap (plan §2.10).

When seeds are available (multi-seed training), each bootstrap iteration
randomly selects one seed to use, so the resulting CI captures both data
and training stochasticity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def draw_seeds(
    seeds: Sequence[int],
    n_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_iter seed values uniformly with replacement from `seeds`.

    Returns an int array of shape (n_iter,).
    """
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty")
    arr = np.asarray(list(seeds), dtype=int)
    return rng.choice(arr, size=n_iter, replace=True)
