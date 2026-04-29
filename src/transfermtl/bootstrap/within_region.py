"""Within-region hierarchical bootstrap (plan §2.10).

For G_ij(r) and Δ_ij(r): cluster-level resampling across regions is not
applicable since the statistic is region-conditional. Instead we resample
scaffolds *within* the target region, then compounds within each scaffold.
Internally this is just `hierarchical_bootstrap` applied to the
region-restricted slice; this thin wrapper exists so call sites read clearly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from transfermtl.bootstrap.hierarchical import hierarchical_bootstrap
from transfermtl.utils.types import BootstrapResult, HierarchicalSamples


def within_region_bootstrap(
    compute_fn: Callable[[HierarchicalSamples], float],
    data: HierarchicalSamples,
    region_ids: np.ndarray,
    target_region: int,
    n_iter: int = 1000,
    level1: Literal["scaffold", "cluster"] = "scaffold",
    seeds: Sequence[int] | None = None,
    save_samples: bool = False,
    rng_seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Hierarchical bootstrap restricted to compounds whose region == target_region."""
    region_arr = np.asarray(region_ids)
    if region_arr.shape[0] != data.values.shape[0]:
        raise ValueError("region_ids must have the same length as data.values")
    mask = region_arr == target_region
    sub = HierarchicalSamples(
        values=data.values[mask],
        scaffold_ids=data.scaffold_ids[mask],
        seed_ids=None if data.seed_ids is None else data.seed_ids[mask],
    )
    return hierarchical_bootstrap(
        compute_fn,
        sub,
        n_iter=n_iter,
        level1=level1,
        seeds=seeds,
        save_samples=save_samples,
        rng_seed=rng_seed,
        alpha=alpha,
    )
