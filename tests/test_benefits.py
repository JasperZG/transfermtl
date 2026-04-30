"""Unit tests for src/transfermtl/benefits/.

Tests use hand-crafted PredictionSchema parquets so they do not depend on
real model training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transfermtl.benefits.aggregate import aggregate_region_benefits
from transfermtl.benefits.delta import compute_region_deltas
from transfermtl.benefits.perf import regional_perf
from transfermtl.utils.schemas import RegionBenefitSchema


def _mock_predictions(
    task: str,
    smiles: list[str],
    y_true: list[float],
    y_pred: list[float],
    seed: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": smiles,
            "task": [task] * len(smiles),
            "y_true": y_true,
            "y_pred": y_pred,
            "seed": [seed] * len(smiles),
        }
    )


# ------------------------------------------------------------------
# regional_perf
# ------------------------------------------------------------------


def test_perf_undefined_for_small_region() -> None:
    """Region with n<30 → NaN."""
    smis = [f"s{i}" for i in range(10)]
    df = _mock_predictions("t1", smis, [0.0, 1.0] * 5, [0.1, 0.9] * 5)
    out = regional_perf(df, smis, "t1", task_type="clf")
    assert np.isnan(out)


def test_perf_undefined_for_single_class() -> None:
    """All-positive region → NaN."""
    smis = [f"s{i}" for i in range(40)]
    df = _mock_predictions("t1", smis, [1.0] * 40, [0.5] * 40)
    out = regional_perf(df, smis, "t1", task_type="clf")
    assert np.isnan(out)


def test_perf_returns_auc_when_valid() -> None:
    """A monotonic predictor returns AUC = 1.0 on a balanced region."""
    n = 40
    smis = [f"s{i}" for i in range(n)]
    y = [0.0] * (n // 2) + [1.0] * (n // 2)
    p = [0.1] * (n // 2) + [0.9] * (n // 2)
    df = _mock_predictions("t1", smis, y, p)
    auc = regional_perf(df, smis, "t1", task_type="clf")
    assert auc == pytest.approx(1.0)


# ------------------------------------------------------------------
# compute_region_deltas
# ------------------------------------------------------------------


def _make_region_predictions(
    region: str,
    n: int,
    task: str,
    seed: int,
    auc: float,
) -> pd.DataFrame:
    """Generate predictions with an approximate target ROC-AUC.

    Two normal distributions whose means are separated proportional to
    `(auc - 0.5)`. Overlapping bell curves yield intermediate AUCs.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    smis = [f"{region}_{i}" for i in range(n)]
    y = [0.0] * half + [1.0] * half
    shift = (abs(auc - 0.5)) * 4.0
    if auc >= 0.5:
        p_neg = rng.normal(loc=-shift, scale=1.0, size=half).tolist()
        p_pos = rng.normal(loc=+shift, scale=1.0, size=half).tolist()
    else:
        p_neg = rng.normal(loc=+shift, scale=1.0, size=half).tolist()
        p_pos = rng.normal(loc=-shift, scale=1.0, size=half).tolist()
    return _mock_predictions(task, smis, y, p_neg + p_pos, seed=seed)


def test_delta_pair_recovers_known() -> None:
    """Region A (aligned) → Δ > 0; Region B (opposed) → Δ < 0."""
    n_per_region = 60
    a_smis = [f"A_{i}" for i in range(n_per_region)]
    b_smis = [f"B_{i}" for i in range(n_per_region)]
    partition = pd.DataFrame(
        {
            "smiles": a_smis + b_smis,
            "region_id": [0] * n_per_region + [1] * n_per_region,
        }
    )

    # Region A: STL ≈ 0.7 AUC; MTL ≈ 0.85 AUC -> Δ_pair > 0.
    # Region B: STL ≈ 0.7 AUC; MTL ≈ 0.55 AUC -> Δ_pair < 0.
    stl_i = pd.concat(
        [
            _make_region_predictions("A", n_per_region, "t1", seed=0, auc=0.7),
            _make_region_predictions("B", n_per_region, "t1", seed=0, auc=0.7),
        ],
        ignore_index=True,
    )
    stl_j = pd.concat(
        [
            _make_region_predictions("A", n_per_region, "t2", seed=0, auc=0.7),
            _make_region_predictions("B", n_per_region, "t2", seed=0, auc=0.7),
        ],
        ignore_index=True,
    )
    mtl_a_i = _make_region_predictions("A", n_per_region, "t1", seed=1, auc=0.85)
    mtl_a_j = _make_region_predictions("A", n_per_region, "t2", seed=1, auc=0.85)
    mtl_b_i = _make_region_predictions("B", n_per_region, "t1", seed=1, auc=0.55)
    mtl_b_j = _make_region_predictions("B", n_per_region, "t2", seed=1, auc=0.55)
    mtl = pd.concat([mtl_a_i, mtl_a_j, mtl_b_i, mtl_b_j], ignore_index=True)

    deltas = compute_region_deltas(stl_i, stl_j, mtl, partition, "t1", "t2", task_type="clf")
    assert deltas[0].delta_pair > 0
    assert deltas[1].delta_pair < 0


def test_task_specific_consistency() -> None:
    """Δ_pair = (Δ_{i←j} + Δ_{j←i})/2 within fp tolerance."""
    n = 40
    smis_a = [f"A_{i}" for i in range(n)]
    partition = pd.DataFrame({"smiles": smis_a, "region_id": [0] * n})

    stl_i = _make_region_predictions("A", n, "t1", seed=0, auc=0.7)
    stl_j = _make_region_predictions("A", n, "t2", seed=0, auc=0.65)
    mtl = pd.concat(
        [
            _make_region_predictions("A", n, "t1", seed=1, auc=0.8),
            _make_region_predictions("A", n, "t2", seed=1, auc=0.55),
        ],
        ignore_index=True,
    )
    d = compute_region_deltas(stl_i, stl_j, mtl, partition, "t1", "t2", task_type="clf")[0]
    assert d.delta_pair == pytest.approx(0.5 * (d.delta_i_from_j + d.delta_j_from_i), abs=1e-9)


# ------------------------------------------------------------------
# aggregate_region_benefits — bootstrap CIs attached
# ------------------------------------------------------------------


def test_bootstrap_ci_attached() -> None:
    """aggregate_region_benefits attaches ci_lo / ci_hi for every region."""
    n = 60
    a_smis = [f"A_{i}" for i in range(n)]
    b_smis = [f"B_{i}" for i in range(n)]
    partition = pd.DataFrame(
        {
            "smiles": a_smis + b_smis,
            "region_id": [0] * n + [1] * n,
        }
    )
    # A1's SplitSchema requires task columns; the bootstrap closure only needs
    # `smiles` and `scaffold` so we hand-craft a minimal frame.
    split = pd.DataFrame(
        {
            "smiles": a_smis + b_smis,
            "scaffold": [f"scaff_A_{i // 5}" for i in range(n)]
            + [f"scaff_B_{i // 5}" for i in range(n)],
        }
    )

    stl_i = pd.concat(
        [
            _make_region_predictions("A", n, "t1", seed=0, auc=0.70),
            _make_region_predictions("B", n, "t1", seed=0, auc=0.70),
        ],
        ignore_index=True,
    )
    stl_j = pd.concat(
        [
            _make_region_predictions("A", n, "t2", seed=0, auc=0.70),
            _make_region_predictions("B", n, "t2", seed=0, auc=0.70),
        ],
        ignore_index=True,
    )
    mtl_a = pd.concat(
        [
            _make_region_predictions("A", n, "t1", seed=1, auc=0.85),
            _make_region_predictions("A", n, "t2", seed=1, auc=0.85),
        ],
        ignore_index=True,
    )
    mtl_b = pd.concat(
        [
            _make_region_predictions("B", n, "t1", seed=1, auc=0.55),
            _make_region_predictions("B", n, "t2", seed=1, auc=0.55),
        ],
        ignore_index=True,
    )
    mtl = pd.concat([mtl_a, mtl_b], ignore_index=True)

    benefits = aggregate_region_benefits(
        stl_preds_i_per_seed={0: stl_i},
        stl_preds_j_per_seed={0: stl_j},
        mtl_preds_per_seed={0: mtl},
        partition=partition,
        split=split,
        task_i="t1",
        task_j="t2",
        task_type="clf",
        n_iter=80,
    )
    RegionBenefitSchema.validate(benefits)
    assert {"ci_lo", "ci_hi"}.issubset(benefits.columns)
    # CI columns populated for at least one region (some may be NaN if region invalid).
    assert benefits[["ci_lo", "ci_hi"]].notna().any().all()
