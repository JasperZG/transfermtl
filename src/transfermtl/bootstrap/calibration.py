"""Pre-pilot bootstrap calibration check (plan §2.10).

Sanity-checks the bootstrap library against three properties before any real
result depends on it. A4 ships the framework; A6/A7 invoke this against real
G_ij(r) data during the pilot.

Checks (relaxed surrogates on the pre-pilot input):
  1. Identity statistic returns CI width < 0.05 (G_ii(r) ≈ 1 in the spec).
  2. A non-trivial mean statistic produces a non-degenerate CI.
  3. ≥80% of bootstrap distributions over a small panel of statistics are
     approximately normal under Shapiro-Wilk (p > alpha_normal).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import shapiro

from transfermtl.bootstrap.hierarchical import hierarchical_bootstrap
from transfermtl.utils.types import HierarchicalSamples


class CalibrationError(RuntimeError):
    """Raised when the §2.10 pre-pilot calibration check fails."""


_PANEL: tuple[Callable[[HierarchicalSamples], float], ...] = (
    lambda d: float(np.mean(d.values)),
    lambda d: float(2.0 * np.mean(d.values) - 1.0),
    lambda d: float(np.mean(d.values) + 0.25),
    lambda d: float(np.mean(np.asarray(d.values, dtype=float) ** 2)),
    lambda d: float(np.mean(np.abs(np.asarray(d.values, dtype=float) - 0.5))),
)


def run_calibration_check(
    synthetic_data: HierarchicalSamples,
    *,
    n_iter: int = 500,
    identity_width_max: float = 0.05,
    normality_alpha: float = 0.01,
    normality_ratio_required: float = 0.80,
    rng_seed: int = 0,
) -> None:
    """Run the §2.10 pre-pilot calibration. Raises CalibrationError on failure."""
    # 1) Identity: must yield a degenerate-tight CI.
    identity = hierarchical_bootstrap(
        lambda _d: 1.0,
        synthetic_data,
        n_iter=n_iter,
        rng_seed=rng_seed,
    )
    width = identity.ci_upper - identity.ci_lower
    if not (width < identity_width_max):
        raise CalibrationError(
            f"identity statistic CI width {width:.4f} is not < {identity_width_max}"
        )

    # 2) Mean: must yield a non-degenerate CI.
    mean_result = hierarchical_bootstrap(
        lambda d: float(np.mean(d.values)),
        synthetic_data,
        n_iter=n_iter,
        save_samples=True,
        rng_seed=rng_seed + 1,
    )
    if not (mean_result.ci_upper > mean_result.ci_lower):
        raise CalibrationError("mean statistic produced a degenerate CI")

    # 3) Shapiro-Wilk normality across the panel.
    n_pass = 0
    for k, fn in enumerate(_PANEL):
        result = hierarchical_bootstrap(
            fn,
            synthetic_data,
            n_iter=n_iter,
            save_samples=True,
            rng_seed=rng_seed + 100 + k,
        )
        assert result.samples is not None  # save_samples=True
        s = result.samples
        s = s[~np.isnan(s)]
        if s.size < 3:
            continue
        if np.std(s) == 0.0:
            n_pass += 1  # constant distribution: trivially "normal".
            continue
        if s.size > 5000:
            s = s[:5000]  # shapiro bounds on N
        try:
            _, p = shapiro(s)
        except ValueError:
            continue
        if p > normality_alpha:
            n_pass += 1

    ratio = n_pass / len(_PANEL)
    if ratio < normality_ratio_required:
        raise CalibrationError(
            f"only {n_pass}/{len(_PANEL)} statistics passed Shapiro-Wilk "
            f"(ratio {ratio:.2f} < {normality_ratio_required})"
        )
