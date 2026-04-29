"""Tests for transfermtl.null (plan §2.13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transfermtl.null import (
    build_null_distribution,
    empirical_pvalue,
    load_null_distribution,
    null_path,
    save_null_distribution,
)


def test_pvalue_extremes() -> None:
    rng = np.random.default_rng(0)
    null = rng.normal(size=200)

    # observed > all → only itself counts in the >= sum: p = 2/(B+1) ≈ 1/(B+1)
    obs_high = float(null.max()) + 1e-9  # strictly above max
    p_high = empirical_pvalue(obs_high, null)
    assert p_high == pytest.approx(1.0 / (1 + len(null)), rel=0.05)

    # observed below all → all null >= observed → p = (1+B)/(1+B) = 1.0
    obs_low = float(null.min()) - 1.0
    p_low = empirical_pvalue(obs_low, null)
    assert p_low == 1.0

    # observed = max(null) → numerator = 1 + #(>=max). For unique max, that is 2.
    obs_max = float(null.max())
    p_max = empirical_pvalue(obs_max, null)
    assert p_max == pytest.approx(2.0 / (1 + len(null)), rel=0.01)


def test_pvalue_smoothing_empty_null() -> None:
    p = empirical_pvalue(0.5, np.array([]))
    assert p == 1.0


def test_pvalue_skips_nan() -> None:
    null = np.array([0.1, np.nan, 0.5, np.nan, 0.7])
    p = empirical_pvalue(0.6, null)
    # Effective null = [0.1, 0.5, 0.7]; #>=0.6 = 1; p = (1+1)/(1+3) = 0.5
    assert p == pytest.approx(0.5)


def test_build_null_distribution_loops_over_partitions() -> None:
    arr = build_null_distribution(
        dataset="tox21",
        statistic="prevalence",
        M=5,
        n_partitions=10,
        compute_statistic=lambda b: 0.1 * b,
    )
    np.testing.assert_allclose(arr, np.linspace(0.0, 0.9, 10))


def test_build_null_distribution_requires_compute_fn() -> None:
    with pytest.raises(NotImplementedError):
        build_null_distribution(
            dataset="tox21",
            statistic="prevalence",
            M=5,
            n_partitions=10,
            compute_statistic=None,
        )


def test_build_null_distribution_rejects_zero_partitions() -> None:
    with pytest.raises(ValueError):
        build_null_distribution(
            dataset="tox21",
            statistic="prevalence",
            M=5,
            n_partitions=0,
            compute_statistic=lambda b: 0.0,
        )


def test_null_io_roundtrip(tmp_path: Path) -> None:
    arr = np.linspace(-1.0, 1.0, 50)
    path = save_null_distribution("tox21", "prevalence", 5, arr, base=tmp_path)
    assert path == null_path("tox21", "prevalence", 5, base=tmp_path)
    back = load_null_distribution("tox21", "prevalence", 5, base=tmp_path)
    np.testing.assert_allclose(back, arr)
