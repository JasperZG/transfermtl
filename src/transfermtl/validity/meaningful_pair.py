"""Meaningful task-pair criterion (plan §2.12).

A task pair (i, j) enters the heterogeneity-prevalence denominator only when
the four §2.12 conditions hold. The pair-level decision is the "denominator
lock" that prevents post-hoc selection bias on prevalence statistics.
"""

from __future__ import annotations

from typing import Literal

from transfermtl.utils.types import BootstrapResult, ValidityFlag
from transfermtl.validity.local_support import (
    REASON_LABEL_BALANCE,
    REASON_N_TEST,
    ValidityConfig,
)

# Public condition tags returned in failed_reasons.
COND_VALID_REGIONS = "c1_valid_regions"
COND_LARGE_DELTA = "c2_large_delta_with_ci"
COND_LABEL_DISTRIBUTION = "c3_label_distribution"
COND_TEST_SIZE = "c4_test_size"


def _ci_excludes_zero(result: BootstrapResult) -> bool:
    return (result.ci_lower > 0.0 and result.ci_upper > 0.0) or (
        result.ci_lower < 0.0 and result.ci_upper < 0.0
    )


def check_meaningful(
    pair_id: str,
    region_validity: dict[int, ValidityFlag],
    region_deltas: dict[int, BootstrapResult],
    cfg: ValidityConfig,
    task_type: Literal["clf", "reg"] = "clf",
) -> tuple[bool, list[str]]:
    """Evaluate the four §2.12 conditions; return (is_meaningful, failed)."""
    del pair_id  # logged by caller; not needed for the rule itself

    failed: list[str] = []

    # c1: ≥3 valid regions per §2.11.
    n_valid = sum(1 for f in region_validity.values() if f.valid)
    if n_valid < cfg.n_min_valid_regions:
        failed.append(COND_VALID_REGIONS)

    # c2: at least one region with |Δ_ij(r)| > epsilon AND CI excluding 0.
    epsilon = cfg.epsilon_clf if task_type == "clf" else cfg.epsilon_reg
    has_large = False
    for r, delta in region_deltas.items():
        # Region must itself be valid; otherwise §2.11 already rejected it.
        if r not in region_validity or not region_validity[r].valid:
            continue
        if abs(delta.estimate) > epsilon and _ci_excludes_zero(delta):
            has_large = True
            break
    if not has_large:
        failed.append(COND_LARGE_DELTA)

    # c3: ≥3 regions with non-degenerate label distributions.
    #     (label_balance subcondition not in failed_reasons; clf only signal —
    #     for regression we count all regions, since label balance is N/A.)
    n_label_ok = sum(
        1 for f in region_validity.values() if REASON_LABEL_BALANCE not in f.failed_reasons
    )
    if n_label_ok < cfg.n_min_valid_regions:
        failed.append(COND_LABEL_DISTRIBUTION)

    # c4: ≥3 regions with adequate test size.
    n_test_ok = sum(1 for f in region_validity.values() if REASON_N_TEST not in f.failed_reasons)
    if n_test_ok < cfg.n_min_valid_regions:
        failed.append(COND_TEST_SIZE)

    return (not failed), failed
