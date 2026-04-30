"""Random-partition null distribution (plan §2.13).

Skeleton in this wave; A6 wires real measurement compute functions in Wave 3.
The signature matches the brief so downstream code can import it now.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def build_null_distribution(
    dataset: str,
    statistic: str,
    M: int,
    n_partitions: int = 200,
    compute_statistic: Callable[[int], float] | None = None,
) -> np.ndarray:
    """Build the null distribution by sweeping random partitions.

    For each random partition index b in [0, n_partitions), compute the
    requested statistic and stack the results. `compute_statistic(b)` is
    expected to load partition b from
    `outputs/partitions/{dataset}/random_b{b}.parquet` and run the same
    pipeline that produced `S^scaffold`. The skeleton enforces the loop
    structure; A6 supplies a concrete `compute_statistic`.
    """
    del dataset, statistic, M  # carried for documentation / call sites
    if compute_statistic is None:
        raise NotImplementedError("compute_statistic must be supplied; A6 wires this in Wave 3")
    if n_partitions <= 0:
        raise ValueError("n_partitions must be positive")
    return np.asarray([float(compute_statistic(b)) for b in range(n_partitions)], dtype=float)
