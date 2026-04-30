"""Tests for transfermtl.validity (plan §2.11-2.12)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from transfermtl.utils.schemas import MeaningfulPairSchema
from transfermtl.utils.types import BootstrapResult, ValidityFlag
from transfermtl.validity import (
    COND_LABEL_DISTRIBUTION,
    COND_LARGE_DELTA,
    COND_TEST_SIZE,
    COND_VALID_REGIONS,
    REASON_CI_WIDTH,
    REASON_GRAD_NORM,
    REASON_LABEL_BALANCE,
    REASON_N_TEST,
    REASON_N_TRAIN,
    ValidityConfig,
    check_local_support,
    check_meaningful,
    write_meaningful_pairs,
)


def _all_pass_kwargs() -> dict:
    return dict(
        n_train_i=100,
        n_train_j=100,
        n_test_i=40,
        n_test_j=40,
        n_test_pos_i=20,
        n_test_neg_i=20,
        n_test_pos_j=20,
        n_test_neg_j=20,
        g_i_norm=1e-3,
        g_j_norm=1e-3,
        g_ij_ci_width=0.2,
        task_type="clf",
        cfg=ValidityConfig(),
    )


def test_local_support_all_pass() -> None:
    flag = check_local_support(**_all_pass_kwargs())
    assert flag.valid is True
    assert flag.failed_reasons == ()


@pytest.mark.parametrize(
    ("override", "expected_reason"),
    [
        ({"n_train_i": 10}, REASON_N_TRAIN),
        ({"n_test_j": 5}, REASON_N_TEST),
        ({"n_test_pos_i": 1}, REASON_LABEL_BALANCE),
        ({"g_ij_ci_width": 0.9}, REASON_CI_WIDTH),
        ({"g_j_norm": 1e-9}, REASON_GRAD_NORM),
    ],
)
def test_local_support_fails_each_condition(override: dict, expected_reason: str) -> None:
    kwargs = _all_pass_kwargs() | override
    flag = check_local_support(**kwargs)
    assert flag.valid is False
    assert expected_reason in flag.failed_reasons


def test_local_support_regression_skips_label_balance() -> None:
    kwargs = _all_pass_kwargs()
    kwargs["task_type"] = "reg"
    # test_min_reg = 50 > test_min_clf = 30; bump test sizes accordingly.
    kwargs["n_test_i"] = 80
    kwargs["n_test_j"] = 80
    kwargs["n_test_pos_i"] = 0  # would fail clf class-balance, irrelevant for reg
    flag = check_local_support(**kwargs)
    assert flag.valid is True
    assert REASON_LABEL_BALANCE not in flag.failed_reasons


def _valid_flag() -> ValidityFlag:
    return ValidityFlag(valid=True, failed_reasons=())


def _invalid_flag(*reasons: str) -> ValidityFlag:
    return ValidityFlag(valid=False, failed_reasons=tuple(reasons))


def _delta_excluding_zero(magnitude: float) -> BootstrapResult:
    sign = 1.0 if magnitude >= 0 else -1.0
    half = max(abs(magnitude) * 0.05, 0.01)
    return BootstrapResult(
        estimate=magnitude,
        ci_lower=sign * (abs(magnitude) - half),
        ci_upper=sign * (abs(magnitude) + half),
    )


def _delta_including_zero(magnitude: float) -> BootstrapResult:
    return BootstrapResult(estimate=magnitude, ci_lower=-abs(magnitude), ci_upper=abs(magnitude))


def test_meaningful_all_conditions_pass() -> None:
    cfg = ValidityConfig()
    region_validity = {r: _valid_flag() for r in range(4)}
    region_deltas = {r: _delta_excluding_zero(2.0) for r in range(4)}
    ok, failed = check_meaningful("p", region_validity, region_deltas, cfg)
    assert ok is True
    assert failed == []


def test_meaningful_pair_4_conditions() -> None:
    cfg = ValidityConfig()

    # c1: too few valid regions (only 2 valid out of needed 3).
    region_validity_c1 = {
        0: _valid_flag(),
        1: _valid_flag(),
        2: _invalid_flag(REASON_GRAD_NORM),
        3: _invalid_flag(REASON_GRAD_NORM),
    }
    region_deltas_c1 = {r: _delta_excluding_zero(2.0) for r in region_validity_c1}
    ok, failed = check_meaningful("p_c1", region_validity_c1, region_deltas_c1, cfg)
    assert ok is False and COND_VALID_REGIONS in failed

    # c2: 3 valid regions, but no region with |Δ| > epsilon AND CI excluding 0.
    region_validity_c2 = {r: _valid_flag() for r in range(3)}
    region_deltas_c2 = {
        0: _delta_including_zero(0.5),  # CI includes 0
        1: _delta_excluding_zero(0.5),  # |Δ| < epsilon (1.5)
        2: _delta_including_zero(2.0),  # CI includes 0 even though large
    }
    ok, failed = check_meaningful("p_c2", region_validity_c2, region_deltas_c2, cfg)
    assert ok is False and COND_LARGE_DELTA in failed

    # c3: <3 regions where label_balance subcondition passed.
    # (Note: c3 is a stricter sub-aspect of validity, so c1 also fails here —
    # the test only asserts COND_LABEL_DISTRIBUTION shows up among the
    # failure reasons.)
    region_validity_c3 = {
        0: _invalid_flag(REASON_LABEL_BALANCE),
        1: _invalid_flag(REASON_LABEL_BALANCE),
        2: _invalid_flag(REASON_LABEL_BALANCE),
        3: _invalid_flag(REASON_LABEL_BALANCE),
    }
    region_deltas_c3 = {r: _delta_excluding_zero(2.0) for r in region_validity_c3}
    ok, failed = check_meaningful("p_c3", region_validity_c3, region_deltas_c3, cfg)
    assert ok is False and COND_LABEL_DISTRIBUTION in failed

    # c4: <3 regions where n_test subcondition passed.
    region_validity_c4 = {
        0: _invalid_flag(REASON_N_TEST),
        1: _invalid_flag(REASON_N_TEST),
        2: _invalid_flag(REASON_N_TEST),
        3: _invalid_flag(REASON_N_TEST),
    }
    region_deltas_c4 = {r: _delta_excluding_zero(2.0) for r in region_validity_c4}
    ok, failed = check_meaningful("p_c4", region_validity_c4, region_deltas_c4, cfg)
    assert ok is False and COND_TEST_SIZE in failed


def test_meaningful_handles_regression_target() -> None:
    cfg = ValidityConfig()
    region_validity = {r: _valid_flag() for r in range(4)}
    # |Δ| = 0.15 RMSE > epsilon_reg (0.10) but < epsilon_clf (1.5)
    region_deltas = {r: _delta_excluding_zero(0.15) for r in range(4)}

    ok_clf, _ = check_meaningful("p", region_validity, region_deltas, cfg, task_type="clf")
    ok_reg, _ = check_meaningful("p", region_validity, region_deltas, cfg, task_type="reg")
    assert ok_clf is False
    assert ok_reg is True


def test_write_meaningful_pairs_roundtrip(tmp_path: Path) -> None:
    results = [
        ("pair_a", True, []),
        ("pair_b", False, [COND_VALID_REGIONS]),
        ("pair_c", False, [COND_LARGE_DELTA, COND_TEST_SIZE]),
    ]
    out = write_meaningful_pairs("tox21", results, base=tmp_path)
    assert out.exists()
    df = pd.read_parquet(out)
    MeaningfulPairSchema.validate(df)
    assert sorted(df["pair_id"].tolist()) == ["pair_a", "pair_b", "pair_c"]
    row_b = df.set_index("pair_id").loc["pair_b"]
    assert row_b["is_meaningful"] is False or row_b["is_meaningful"] == 0  # bool cast
    assert COND_VALID_REGIONS in list(row_b["failed_reasons"])
