"""Two-level hierarchical bootstrap (plan §2.10).

Resamples scaffolds (level 1) with replacement, then compounds within each
resampled scaffold (level 2). Optionally mixes across training seeds at the
iteration level so the resulting CI captures both data and training
stochasticity.

Plain iid bootstrap is incorrect for this project: compounds within a
scaffold are not independent, so naive resampling under-estimates variance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from transfermtl.bootstrap.percentile import percentile_ci
from transfermtl.bootstrap.seed_mixing import draw_seeds
from transfermtl.utils.types import BootstrapResult, HierarchicalSamples


def _index_by_scaffold(scaffold_ids: np.ndarray) -> tuple[np.ndarray, dict[object, np.ndarray]]:
    """Return (unique_scaffolds, {scaffold_id: row_indices})."""
    unique = np.unique(scaffold_ids)
    groups = {s: np.where(scaffold_ids == s)[0] for s in unique}
    return unique, groups


def _slice_samples(data: HierarchicalSamples, idx: np.ndarray) -> HierarchicalSamples:
    return HierarchicalSamples(
        values=data.values[idx],
        scaffold_ids=data.scaffold_ids[idx],
        seed_ids=None if data.seed_ids is None else data.seed_ids[idx],
    )


def hierarchical_bootstrap(
    compute_fn: Callable[[HierarchicalSamples], float],
    data: HierarchicalSamples,
    n_iter: int = 1000,
    level1: Literal["scaffold", "cluster"] = "scaffold",
    seeds: Sequence[int] | None = None,
    save_samples: bool = False,
    rng_seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Compute a hierarchical-bootstrap CI for `compute_fn(data)`.

    Algorithm per plan §2.10:
      1. (Optional) Pick one seed for this iteration; restrict data to it.
      2. Resample scaffolds (level-1 clusters) with replacement.
      3. For each resampled scaffold, resample compounds within it with
         replacement.
      4. Apply `compute_fn` to the resampled HierarchicalSamples.

    `level1` is accepted for documentation / future use; resampling always
    operates over `data.scaffold_ids` (callers swap in cluster IDs there for
    latent partitioning).
    """
    del level1  # informational only — scaffold_ids carries the cluster identity

    if data.values.shape[0] != data.scaffold_ids.shape[0]:
        raise ValueError("values and scaffold_ids must have the same length")
    if data.seed_ids is not None and data.seed_ids.shape[0] != data.values.shape[0]:
        raise ValueError("seed_ids must have the same length as values")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")

    rng = np.random.default_rng(rng_seed)

    use_seed_mixing = seeds is not None and data.seed_ids is not None
    if use_seed_mixing:
        assert seeds is not None  # type narrowing for mypy
        chosen_seeds = draw_seeds(seeds, n_iter, rng)
    else:
        chosen_seeds = np.zeros(n_iter, dtype=int)

    samples = np.empty(n_iter, dtype=float)

    for b in range(n_iter):
        if use_seed_mixing:
            seed_mask = data.seed_ids == chosen_seeds[b]
            seed_idx = np.where(seed_mask)[0]
            if seed_idx.size == 0:
                samples[b] = np.nan
                continue
            sub = _slice_samples(data, seed_idx)
        else:
            sub = data

        unique, groups = _index_by_scaffold(sub.scaffold_ids)
        if unique.size == 0:
            samples[b] = np.nan
            continue

        picked = rng.choice(unique, size=unique.size, replace=True)

        chunks: list[np.ndarray] = []
        for s in picked:
            members = groups[s]
            if members.size == 0:
                continue
            resampled = rng.choice(members, size=members.size, replace=True)
            chunks.append(resampled)

        if not chunks:
            samples[b] = np.nan
            continue

        all_idx = np.concatenate(chunks)
        boot = _slice_samples(sub, all_idx)
        samples[b] = float(compute_fn(boot))

    estimate = float(compute_fn(data))
    ci_lo, ci_hi = percentile_ci(samples, alpha=alpha)

    return BootstrapResult(
        estimate=estimate,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        samples=samples if save_samples else None,
    )


def bootstrap_result_to_dict(
    result: BootstrapResult, include_samples: bool = False
) -> dict[str, object]:
    """JSON-friendly view of a BootstrapResult.

    `samples` (np.ndarray) is flattened to a list when `include_samples=True`,
    otherwise dropped. Used by callers that want to log a result.
    """
    out: dict[str, object] = {
        "estimate": result.estimate,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
    }
    if include_samples and result.samples is not None:
        out["samples"] = result.samples.flatten().tolist()
    return out
