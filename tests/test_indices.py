"""Unit tests for src/transfermtl/indices/."""

from __future__ import annotations

import numpy as np
import pytest

from transfermtl.indices.cancellation import compute_C
from transfermtl.indices.heterogeneity_index import compute_H
from transfermtl.indices.sign_heterogeneity import (
    compute_S_pair,
    compute_S_task_specific,
)
from transfermtl.utils.types import BootstrapResult


def _br(estimate: float, lo: float, hi: float) -> BootstrapResult:
    return BootstrapResult(estimate=estimate, ci_lower=lo, ci_upper=hi, samples=None)


# ------------------------------------------------------------------
# Sign heterogeneity
# ------------------------------------------------------------------


def test_S_synthetic_fixture_true() -> None:
    """Synthetic fixture: aligned region → +Δ, opposed region → −Δ. S_12 = True."""
    deltas = {
        0: _br(estimate=3.0, lo=2.0, hi=4.0),  # +, CI excludes 0
        1: _br(estimate=-2.5, lo=-3.5, hi=-1.5),  # -, CI excludes 0
    }
    assert compute_S_pair(deltas, epsilon=1.5) is True


def test_S_requires_ci_excludes_zero() -> None:
    """Large |Δ| but CI crosses 0 → S = False."""
    deltas = {
        0: _br(estimate=3.0, lo=-1.0, hi=7.0),  # CI includes 0 → invalid positive
        1: _br(estimate=-2.5, lo=-3.5, hi=-1.5),
    }
    assert compute_S_pair(deltas, epsilon=1.5) is False


def test_S_requires_both_signs() -> None:
    """All-positive deltas → S = False."""
    deltas = {
        0: _br(estimate=2.0, lo=1.0, hi=3.0),
        1: _br(estimate=3.0, lo=2.0, hi=4.0),
    }
    assert compute_S_pair(deltas) is False


def test_S_task_specific_returns_two_flags() -> None:
    deltas_i = {
        0: _br(2.0, 1.0, 3.0),
        1: _br(-2.5, -3.5, -1.5),
    }
    deltas_j = {
        0: _br(2.0, 1.0, 3.0),
        1: _br(2.5, 1.5, 3.5),  # all positive
    }
    s_i, s_j = compute_S_task_specific(deltas_i, deltas_j, epsilon=1.5)
    assert s_i is True
    assert s_j is False


# ------------------------------------------------------------------
# Heterogeneity index H
# ------------------------------------------------------------------


def test_H_zero_for_uniform_deltas() -> None:
    """Equal Δ across regions → H = 0."""
    deltas = {0: 2.0, 1: 2.0, 2: 2.0}
    sizes = {0: 50, 1: 80, 2: 30}
    assert compute_H(deltas, sizes) == pytest.approx(0.0)


def test_H_grows_with_spread() -> None:
    """Larger spread of regional Δ → larger H."""
    sizes = {0: 50, 1: 50}
    h_low = compute_H({0: 1.0, 1: 1.5}, sizes)
    h_high = compute_H({0: -3.0, 1: 3.0}, sizes)
    assert h_high > h_low


# ------------------------------------------------------------------
# Cancellation index C
# ------------------------------------------------------------------


def test_C_large_for_sign_flips() -> None:
    """Synthetic fixture: Δ ≈ +3 in region A, -3 in region B → C very large."""
    deltas = {0: 3.0, 1: -3.0}
    sizes = {0: 100, 1: 100}
    c = compute_C(deltas, sizes, eta=0.5)
    assert c > 5.0


def test_C_one_when_no_cancellation() -> None:
    """Uniform-sign deltas → numerator ≈ |signed sum|; with η=0.5, C slightly < 1."""
    deltas = {0: 2.0, 1: 2.0, 2: 2.0}
    sizes = {0: 100, 1: 100, 2: 100}
    c = compute_C(deltas, sizes, eta=0.5)
    # Numerically: 600 / (600 + 0.5) = 0.999.. — close to 1 from below.
    assert 0.95 < c < 1.0
    assert c == pytest.approx(600 / 600.5, abs=1e-9)


def test_indices_handle_nan() -> None:
    """NaN regional Δ does not crash and is skipped."""
    deltas = {0: 2.0, 1: float("nan"), 2: -3.0}
    sizes = {0: 50, 1: 50, 2: 50}
    h = compute_H(deltas, sizes)
    c = compute_C(deltas, sizes)
    assert not np.isnan(h)
    assert not np.isnan(c)
