"""Tests for transfermtl.bootstrap (plan §2.10).

Covers:
- Mean recovery and CI coverage on synthetic data with known mean
- Cluster-respecting resampling (level-1 picks scaffolds, level-2 stays inside)
- Uniform seed mixing across many iterations
- Calibration check passes on the A1 synthetic fixture
- BootstrapResult is JSON-serializable (with samples flattened)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from transfermtl.bootstrap import (
    CalibrationError,
    bootstrap_result_to_dict,
    draw_seeds,
    hierarchical_bootstrap,
    percentile_ci,
    run_calibration_check,
    within_region_bootstrap,
)
from transfermtl.utils.types import BootstrapResult, HierarchicalSamples


def _make_clustered_normal(
    *,
    n_scaffolds: int = 30,
    per_scaffold: int = 10,
    mean: float = 1.5,
    sd: float = 0.5,
    rng_seed: int = 0,
) -> tuple[HierarchicalSamples, float]:
    rng = np.random.default_rng(rng_seed)
    values = rng.normal(loc=mean, scale=sd, size=n_scaffolds * per_scaffold)
    scaffold_ids = np.repeat(np.arange(n_scaffolds), per_scaffold)
    return (
        HierarchicalSamples(values=values, scaffold_ids=scaffold_ids),
        float(np.mean(values)),
    )


def test_bootstrap_recovers_known_mean() -> None:
    data, sample_mean = _make_clustered_normal()
    result = hierarchical_bootstrap(
        lambda d: float(np.mean(d.values)),
        data,
        n_iter=1000,
        rng_seed=0,
    )
    assert abs(result.estimate - sample_mean) < 1e-9  # estimate is point on data
    # Bootstrap mean of the bootstrap samples should also be near sample_mean
    # (this exercises the resampling distribution).
    midpoint = 0.5 * (result.ci_lower + result.ci_upper)
    assert abs(midpoint - sample_mean) < 0.05


def test_bootstrap_ci_coverage() -> None:
    """Coverage of the 95% CI: ≥93/100 trials should contain the true mean."""
    true_mean = 0.7
    n_hits = 0
    n_trials = 100
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        n_scaffolds = 25
        per_scaffold = 8
        values = rng.normal(loc=true_mean, scale=0.4, size=n_scaffolds * per_scaffold)
        scaffold_ids = np.repeat(np.arange(n_scaffolds), per_scaffold)
        data = HierarchicalSamples(values=values, scaffold_ids=scaffold_ids)
        result = hierarchical_bootstrap(
            lambda d: float(np.mean(d.values)),
            data,
            n_iter=300,
            rng_seed=trial,
        )
        if result.ci_lower <= true_mean <= result.ci_upper:
            n_hits += 1
    assert n_hits >= 93, f"coverage {n_hits}/100 below 93"


def test_bootstrap_matches_normal_clt_approximation() -> None:
    """Acceptance: on iid normal data, bootstrap CI matches mean ± 1.96·σ/√n within 0.02.

    Set scaffold_ids = arange(n) so every compound is its own scaffold; level-1
    resampling becomes ordinary iid bootstrap, level-2 is a no-op, and the CI
    converges to the CLT prediction.
    """
    rng = np.random.default_rng(42)
    n = 1000
    values = rng.normal(loc=0.0, scale=1.0, size=n)
    scaffold_ids = np.arange(n)
    data = HierarchicalSamples(values=values, scaffold_ids=scaffold_ids)
    result = hierarchical_bootstrap(
        lambda d: float(np.mean(d.values)),
        data,
        n_iter=1000,
        rng_seed=7,
    )
    expected_half = 1.96 * np.std(values, ddof=0) / np.sqrt(n)
    expected_lo = float(np.mean(values)) - expected_half
    expected_hi = float(np.mean(values)) + expected_half
    assert abs(result.ci_lower - expected_lo) < 0.02
    assert abs(result.ci_upper - expected_hi) < 0.02


def test_hierarchical_resample_respects_clusters() -> None:
    """When scaffold s is picked at level 1, only its compounds appear in the slice."""
    n_scaffolds = 6
    per_scaffold = 4
    values = np.arange(n_scaffolds * per_scaffold, dtype=float)
    scaffold_ids = np.repeat(np.arange(n_scaffolds), per_scaffold)
    data = HierarchicalSamples(values=values, scaffold_ids=scaffold_ids)

    membership = {
        s: set(np.where(scaffold_ids == s)[0].tolist()) for s in range(n_scaffolds)
    }

    def compute_fn(boot: HierarchicalSamples) -> float:
        # Every value-index (recovered from values' integer encoding) must
        # belong to one of the picked scaffolds.
        picked = set(np.unique(boot.scaffold_ids).tolist())
        for v, s in zip(boot.values.astype(int), boot.scaffold_ids.astype(int), strict=True):
            assert int(s) in picked
            assert int(v) in membership[int(s)]
        return float(np.mean(boot.values))

    hierarchical_bootstrap(compute_fn, data, n_iter=200, rng_seed=11)


def test_seed_mixing_uniform() -> None:
    seeds = [10, 20, 30, 40, 50]
    rng = np.random.default_rng(0)
    drawn = draw_seeds(seeds, n_iter=10_000, rng=rng)
    counts = {s: int(np.sum(drawn == s)) for s in seeds}
    for s, c in counts.items():
        assert 1500 <= c <= 2500, f"seed {s} drawn {c} times outside [1500, 2500]"


def test_seed_mixing_in_bootstrap() -> None:
    """Plumbing: providing seeds + seed_ids drives the seed-mask path."""
    rng = np.random.default_rng(0)
    n_scaffolds = 10
    per_scaffold = 6
    n = n_scaffolds * per_scaffold
    values = rng.normal(size=n * 3)
    scaffold_ids = np.tile(np.repeat(np.arange(n_scaffolds), per_scaffold), 3)
    seed_ids = np.repeat([0, 1, 2], n)
    data = HierarchicalSamples(values=values, scaffold_ids=scaffold_ids, seed_ids=seed_ids)

    seen_seeds: set[int] = set()

    def compute_fn(boot: HierarchicalSamples) -> float:
        assert boot.seed_ids is not None
        seen_seeds.update(int(s) for s in np.unique(boot.seed_ids).tolist())
        return float(np.mean(boot.values))

    result = hierarchical_bootstrap(
        compute_fn,
        data,
        n_iter=200,
        seeds=[0, 1, 2],
        rng_seed=3,
    )
    assert seen_seeds == {0, 1, 2}, seen_seeds
    assert result.ci_lower < result.ci_upper


def test_calibration_check_passes_on_synthetic(synthetic_combined: pd.DataFrame) -> None:
    df = synthetic_combined
    scaffold_ids = pd.factorize(df["scaffold"])[0]
    data = HierarchicalSamples(
        values=df["task_1"].to_numpy(dtype=float),
        scaffold_ids=scaffold_ids,
    )
    run_calibration_check(data, n_iter=400, rng_seed=0)


def test_calibration_check_raises_on_pathological() -> None:
    """Single-scaffold, all-identical data should fail the mean-CI non-degeneracy check."""
    values = np.zeros(40)
    scaffold_ids = np.zeros(40, dtype=int)
    data = HierarchicalSamples(values=values, scaffold_ids=scaffold_ids)
    with pytest.raises(CalibrationError):
        run_calibration_check(data, n_iter=200)


def test_save_samples_round_trip() -> None:
    data, _ = _make_clustered_normal()
    result = hierarchical_bootstrap(
        lambda d: float(np.mean(d.values)),
        data,
        n_iter=300,
        save_samples=True,
        rng_seed=0,
    )
    serial = bootstrap_result_to_dict(result, include_samples=True)
    payload = json.dumps(serial)
    back = json.loads(payload)
    assert "samples" in back
    assert len(back["samples"]) == 300
    assert back["estimate"] == result.estimate


def test_bootstrap_result_is_json_serializable_without_samples() -> None:
    """Acceptance: BootstrapResult is JSON-serializable when samples are dropped."""
    r = BootstrapResult(estimate=0.1, ci_lower=-0.2, ci_upper=0.3, samples=None)
    json.dumps(bootstrap_result_to_dict(r))


def test_percentile_ci_quantiles() -> None:
    samples = np.linspace(0.0, 1.0, 1001)
    lo, hi = percentile_ci(samples, alpha=0.05)
    assert abs(lo - 0.025) < 0.01
    assert abs(hi - 0.975) < 0.01


def test_percentile_ci_handles_nan_and_empty() -> None:
    samples = np.array([np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
    lo, hi = percentile_ci(samples, alpha=0.5)
    # Effective samples = [0, 1, 2, 3, 4]; quartiles = (1.0, 3.0).
    assert lo == 1.0 and hi == 3.0
    lo2, hi2 = percentile_ci(np.array([np.nan, np.nan]))
    assert np.isnan(lo2) and np.isnan(hi2)


def test_within_region_filters_to_target() -> None:
    rng = np.random.default_rng(0)
    n = 60
    values = rng.normal(size=n)
    scaffold_ids = np.repeat(np.arange(10), n // 10)
    region_ids = np.repeat([0, 1], n // 2)
    # Make region 0 mean = 5 (very different from region 1 mean ~ 0).
    values[region_ids == 0] += 5.0
    data = HierarchicalSamples(values=values, scaffold_ids=scaffold_ids)

    seen_regions: set[int] = set()

    def compute_fn(boot: HierarchicalSamples) -> float:
        # We can't see region_ids inside compute_fn, but the mean magnitude
        # tells us we are restricted to region 0.
        return float(np.mean(boot.values))

    result = within_region_bootstrap(
        compute_fn,
        data,
        region_ids=region_ids,
        target_region=0,
        n_iter=200,
        rng_seed=0,
    )
    assert result.estimate > 4.0
    del seen_regions  # keep linters happy without changing structure


def test_n_iter_must_be_positive() -> None:
    data, _ = _make_clustered_normal()
    with pytest.raises(ValueError):
        hierarchical_bootstrap(lambda d: 0.0, data, n_iter=0)
