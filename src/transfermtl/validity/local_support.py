"""Local-support validity criterion (plan §2.11).

A regional gradient-affinity estimate G_ij(r) is *valid* iff all five
conditions below hold. Any failure is recorded by name in
ValidityFlag.failed_reasons; check_meaningful (§2.12) reads those names back
out to evaluate per-pair conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transfermtl.utils.types import ValidityFlag


@dataclass(frozen=True)
class ValidityConfig:
    """Thresholds from configs/_shared/preprocess.yaml (plan §2.21)."""

    n_min: int = 50
    test_min_clf: int = 30
    test_min_reg: int = 50
    test_min_pos: int = 5
    test_min_neg: int = 5
    ci_width_max: float = 0.4
    grad_norm_min: float = 1.0e-6
    epsilon_clf: float = 1.5
    epsilon_reg: float = 0.10
    n_min_valid_regions: int = 3


# Public reason tags. Order in failed_reasons follows the order checks fire.
REASON_N_TRAIN = "n_train"
REASON_N_TEST = "n_test"
REASON_LABEL_BALANCE = "label_balance"
REASON_CI_WIDTH = "ci_width"
REASON_GRAD_NORM = "grad_norm"


def check_local_support(
    *,
    n_train_i: int,
    n_train_j: int,
    n_test_i: int,
    n_test_j: int,
    n_test_pos_i: int,
    n_test_neg_i: int,
    n_test_pos_j: int,
    n_test_neg_j: int,
    g_i_norm: float,
    g_j_norm: float,
    g_ij_ci_width: float,
    task_type: Literal["clf", "reg"],
    cfg: ValidityConfig,
) -> ValidityFlag:
    """Evaluate the five §2.11 validity conditions for one (pair, region)."""
    failed: list[str] = []

    # 1. Local support (training).
    if min(n_train_i, n_train_j) < cfg.n_min:
        failed.append(REASON_N_TRAIN)

    # 2. Test-set size.
    test_min = cfg.test_min_clf if task_type == "clf" else cfg.test_min_reg
    if min(n_test_i, n_test_j) < test_min:
        failed.append(REASON_N_TEST)

    # 3. Class balance (clf only).
    if task_type == "clf" and (
        n_test_pos_i < cfg.test_min_pos
        or n_test_neg_i < cfg.test_min_neg
        or n_test_pos_j < cfg.test_min_pos
        or n_test_neg_j < cfg.test_min_neg
    ):
        failed.append(REASON_LABEL_BALANCE)

    # 4. CI width on G_ij(r).
    if g_ij_ci_width > cfg.ci_width_max:
        failed.append(REASON_CI_WIDTH)

    # 5. Non-zero gradients.
    if min(g_i_norm, g_j_norm) <= cfg.grad_norm_min:
        failed.append(REASON_GRAD_NORM)

    return ValidityFlag(valid=not failed, failed_reasons=tuple(failed))
