"""Tests for A7: pilot pair selection, criteria evaluation, decision rendering.

The full end-to-end smoke test (`test_pilot_smoke_synthetic`) requires A6's
gradients/benefits/indices modules to be present; until A6 lands it is marked
``skip`` rather than silently fabricating measurements that mask integration
bugs. The deferral is also recorded in the wave3_a7_complete manifest.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transfermtl.predictor.criteria import (
    Decision,
    PhenomenonResult,
    PredictorResult,
    decide,
    evaluate_phenomenon_criteria,
    evaluate_predictor_criteria,
)
from transfermtl.predictor.pilot_baseline import (
    cross_seed_sign_agreement,
    evaluate_pilot_predictor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


def test_select_pilot_pairs_deterministic() -> None:
    sel = _load_script("select_pilot_pairs")
    a = sel.select_pilot_pairs(seed=42)
    b = sel.select_pilot_pairs(seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_select_pilot_pairs_seed_changes_output() -> None:
    sel = _load_script("select_pilot_pairs")
    a = sel.select_pilot_pairs(seed=1)
    b = sel.select_pilot_pairs(seed=2)
    # Same shape, different content (extremely high-prob across seeds).
    assert a.shape == b.shape
    assert not a.equals(b)


def test_select_pilot_pairs_categories() -> None:
    sel = _load_script("select_pilot_pairs")
    df = sel.select_pilot_pairs(seed=42)
    cats = set(df["category"].unique())
    assert {"within_nr", "within_sr", "cross_mechanism", "cross_soc"}.issubset(cats)
    assert (df["dataset"] == "tox21").sum() == 8 + 5 + 10 + 2
    assert (df["dataset"] == "sider").sum() == 25


# ---------------------------------------------------------------------------
# Phenomenon criteria
# ---------------------------------------------------------------------------


def _phen(**kw) -> PhenomenonResult:
    base = dict(
        n_pairs_with_sign_het=10,
        n_meaningful_pairs=30,
        pct_meaningful_with_sign_het=0.30,
        observed_prevalence=0.20,
        null_percentile_threshold=0.10,
        null_p_value=0.001,
    )
    base.update(kw)
    return PhenomenonResult(**base)


def test_phenomenon_criteria_all_pass() -> None:
    crits = evaluate_phenomenon_criteria(_phen())
    assert all(c.passed for c in crits)
    assert [c.name for c in crits] == [
        "phenomenon_1_n_pairs_with_sign_het",
        "phenomenon_2_pct_meaningful",
        "phenomenon_3_null_test",
    ]


@pytest.mark.parametrize(
    "kwargs,failing",
    [
        ({"n_pairs_with_sign_het": 4}, "phenomenon_1_n_pairs_with_sign_het"),
        ({"pct_meaningful_with_sign_het": 0.10}, "phenomenon_2_pct_meaningful"),
        ({"observed_prevalence": 0.05}, "phenomenon_3_null_test"),
    ],
)
def test_phenomenon_criteria_each_failure(kwargs: dict, failing: str) -> None:
    crits = evaluate_phenomenon_criteria(_phen(**kwargs))
    failed_names = {c.name for c in crits if not c.passed}
    assert failing in failed_names


# ---------------------------------------------------------------------------
# Predictor criteria
# ---------------------------------------------------------------------------


def _pred(**kw) -> PredictorResult:
    base = dict(
        auroc_g_ij=0.78,
        auroc_best_baseline=0.65,
        best_baseline_name="scaffold_tanimoto",
        spearman_rho=0.55,
        cross_seed_agreement=0.85,
    )
    base.update(kw)
    return PredictorResult(**base)


def test_predictor_criteria_all_pass() -> None:
    crits = evaluate_predictor_criteria(_pred())
    assert all(c.passed for c in crits)


@pytest.mark.parametrize(
    "kwargs,failing",
    [
        ({"auroc_g_ij": 0.65}, "predictor_1_auroc_floor"),
        ({"auroc_best_baseline": 0.74}, "predictor_2_baseline_lift"),
        ({"spearman_rho": 0.20}, "predictor_3_spearman"),
        ({"cross_seed_agreement": 0.50}, "predictor_4_cross_seed_agreement"),
    ],
)
def test_predictor_criteria_each_failure(kwargs: dict, failing: str) -> None:
    crits = evaluate_predictor_criteria(_pred(**kwargs))
    failed = {c.name for c in crits if not c.passed}
    assert failing in failed


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


def test_decide_proceed_when_all_pass() -> None:
    phen = evaluate_phenomenon_criteria(_phen())
    pred = evaluate_predictor_criteria(_pred())
    assert decide(phen, pred) == Decision.PROCEED


def test_decide_workshop_when_predictor_floor_fails() -> None:
    phen = evaluate_phenomenon_criteria(_phen())
    pred = evaluate_predictor_criteria(_pred(auroc_g_ij=0.65))
    assert decide(phen, pred) == Decision.WORKSHOP


def test_decide_investigate_when_floor_passes_but_other_fails() -> None:
    phen = evaluate_phenomenon_criteria(_phen())
    pred = evaluate_predictor_criteria(_pred(spearman_rho=0.10))
    assert decide(phen, pred) == Decision.INVESTIGATE


def test_decide_pivot_when_phenomenon_fails_and_predictor_untestable() -> None:
    phen = evaluate_phenomenon_criteria(_phen(n_pairs_with_sign_het=1))
    assert decide(phen, None) == Decision.PIVOT


def test_decide_drop_when_both_fail() -> None:
    phen = evaluate_phenomenon_criteria(_phen(n_pairs_with_sign_het=1))
    pred = evaluate_predictor_criteria(_pred(auroc_g_ij=0.5, spearman_rho=0.0))
    assert decide(phen, pred) == Decision.DROP


# ---------------------------------------------------------------------------
# Pilot predictor (cross-seed agreement + AUROC pipeline)
# ---------------------------------------------------------------------------


def test_cross_seed_sign_agreement_unanimous() -> None:
    df = pd.DataFrame(
        {
            "pair_id": ["p1"] * 6 + ["p2"] * 6,
            "region_id": [0, 0, 0, 1, 1, 1] * 2,
            "seed": [0, 1, 2, 0, 1, 2] * 2,
            "G_ij": [0.8, 0.7, 0.9, -0.5, -0.6, -0.4, 0.3, -0.2, 0.4, -0.5, 0.6, -0.7],
        }
    )
    # Pair 1: both regions unanimous → 100% agreement on cells.
    # Pair 2: signs flip across seeds → 0% agreement on cells.
    # Mean over all 4 cells = 0.5.
    assert cross_seed_sign_agreement(df) == pytest.approx(0.5)


def test_evaluate_pilot_predictor_auroc_and_spearman() -> None:
    rng = np.random.default_rng(0)
    n = 200
    g = rng.normal(size=n)
    delta = g + rng.normal(scale=0.3, size=n)  # strong positive correlation
    baseline = rng.normal(size=n)  # noise
    df = pd.DataFrame(
        {
            "pair_id": [f"p{i // 5}" for i in range(n)],
            "region_id": [(i % 5) for i in range(n)],
            "seed": [(i % 3) for i in range(n)],
            "G_ij": g,
            "delta_pair": delta,
            "scaffold_tanimoto": baseline,
            "embedding_distance": baseline + rng.normal(scale=0.5, size=n),
            "label_correlation": baseline,
        }
    )
    scores = evaluate_pilot_predictor(df)
    assert scores.auroc_g_ij > 0.85
    assert scores.spearman_rho > 0.85
    assert scores.auroc_g_ij > scores.auroc_best_baseline


# ---------------------------------------------------------------------------
# pilot_decision.py end-to-end (via mocked JSON)
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_pilot_decision_script_runs(tmp_path: Path) -> None:
    pd_script = _load_script("pilot_decision")

    phen_path = tmp_path / "phenomenon.json"
    pred_path = tmp_path / "predictor.json"
    per_ds_path = tmp_path / "per_dataset.json"
    examples_path = tmp_path / "examples.json"

    _write_json(
        phen_path,
        dict(
            n_pairs_with_sign_het=8,
            n_meaningful_pairs=25,
            pct_meaningful_with_sign_het=0.32,
            observed_prevalence=0.18,
            null_percentile_threshold=0.10,
            null_p_value=0.005,
        ),
    )
    _write_json(
        pred_path,
        dict(
            auroc_g_ij=0.78,
            auroc_best_baseline=0.66,
            best_baseline_name="scaffold_tanimoto",
            spearman_rho=0.50,
            cross_seed_agreement=0.83,
        ),
    )
    _write_json(
        per_ds_path,
        [
            {
                "dataset": "tox21",
                "n_pairs": 25,
                "n_meaningful": 18,
                "n_sign_het": 5,
                "prevalence": 0.28,
            },
            {
                "dataset": "sider",
                "n_pairs": 25,
                "n_meaningful": 7,
                "n_sign_het": 3,
                "prevalence": 0.43,
            },
        ],
    )
    _write_json(
        examples_path,
        [
            {
                "dataset": "tox21",
                "task_i": "NR-AR",
                "task_j": "SR-MMP",
                "label": "cross_mechanism",
                "delta_min": -2.4,
                "delta_max": 3.1,
                "sign_het": True,
                "c_ij": 6.2,
                "h_ij": 2.7,
            },
            {
                "dataset": "tox21",
                "task_i": "NR-AR",
                "task_j": "NR-AR-LBD",
                "label": "within_nr",
                "delta_min": 0.5,
                "delta_max": 2.8,
                "sign_het": False,
                "c_ij": 1.1,
                "h_ij": 1.1,
            },
            {
                "dataset": "sider",
                "task_i": "Hepatobiliary disorders",
                "task_j": "Skin and subcutaneous tissue disorders",
                "label": "cross_soc",
                "delta_min": -1.9,
                "delta_max": 2.0,
                "sign_het": True,
                "c_ij": 4.0,
                "h_ij": 1.8,
            },
        ],
    )

    out_md = tmp_path / "decision.md"
    out_summary = tmp_path / "summary.parquet"
    rc = pd_script.main(
        [
            "--phenomenon-json",
            str(phen_path),
            "--predictor-json",
            str(pred_path),
            "--per-dataset-json",
            str(per_ds_path),
            "--examples-json",
            str(examples_path),
            "--out-md",
            str(out_md),
            "--out-summary",
            str(out_summary),
        ]
    )
    assert rc == 0
    text = out_md.read_text()
    assert "PROCEED" in text
    assert "phenomenon_1_n_pairs_with_sign_het" in text
    assert "predictor_1_auroc_floor" in text
    assert "Decision tree (plan §5.8)" in text

    summary = pd.read_parquet(out_summary)
    assert (summary["block"] == "phenomenon").sum() == 3
    assert (summary["block"] == "predictor").sum() == 4
    assert (summary["block"] == "decision").sum() == 1
    assert summary[summary["block"] == "decision"]["detail"].iloc[0] == "PROCEED"


def test_pilot_decision_script_handles_untestable_predictor(tmp_path: Path) -> None:
    pd_script = _load_script("pilot_decision")
    phen_path = tmp_path / "phen.json"
    _write_json(
        phen_path,
        dict(
            n_pairs_with_sign_het=2,
            n_meaningful_pairs=5,
            pct_meaningful_with_sign_het=0.0,
            observed_prevalence=0.05,
            null_percentile_threshold=0.10,
            null_p_value=0.5,
        ),
    )
    out_md = tmp_path / "decision.md"
    out_summary = tmp_path / "summary.parquet"
    rc = pd_script.main(
        [
            "--phenomenon-json",
            str(phen_path),
            "--out-md",
            str(out_md),
            "--out-summary",
            str(out_summary),
        ]
    )
    assert rc == 0
    text = out_md.read_text()
    assert "PIVOT" in text
    assert "untestable" in text.lower()


# ---------------------------------------------------------------------------
# Full e2e smoke test — exercises A2-fixture, A3 training, A6 measurement, A7 indices
# ---------------------------------------------------------------------------


def _have_a6_modules() -> bool:
    try:
        importlib.import_module("transfermtl.gradients.extract")
        importlib.import_module("transfermtl.benefits.delta")
        importlib.import_module("transfermtl.indices.sign_heterogeneity")
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _have_a6_modules(),
    reason="A6 (gradients/benefits/indices) has not landed yet — pilot e2e smoke deferred",
)
def test_pilot_smoke_synthetic(
    tmp_path: Path,
    synthetic_dataset: pd.DataFrame,
    synthetic_partition: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end pilot pipeline on the synthetic fixture.

    1. Short STL on task_1, task_2; short pairwise MTL on (task_1, task_2)
    2. A6 regional gradient extraction → assert G_12(A) > 0, G_12(B) < 0
    3. A6 regional benefits → Δ_pair per region
    4. A7 sign-heterogeneity index → assert S_12 = True

    Wall-clock target: <60s on CPU (plan/A7 acceptance criterion).
    """
    import time

    from tests.synthetic_fixture.featurize import build_synthetic_loader
    from transfermtl.benefits import perf as perf_mod
    from transfermtl.benefits.delta import compute_region_deltas
    from transfermtl.gradients.affinity import cosine_affinity
    from transfermtl.gradients.extract import compute_regional_gradient
    from transfermtl.indices.sign_heterogeneity import compute_S_pair
    from transfermtl.training import (
        TrainConfig,
        train_pairwise_mtl,
        train_stl,
    )
    from transfermtl.utils.types import BootstrapResult

    # The 200-mol synthetic fixture has ~15 test compounds per region —
    # regional_perf's production size floor is 30, so without relaxation it
    # returns NaN here. Production use on Tox21 / SIDER comfortably clears the
    # 30-compound floor; the relaxation is fixture-only.
    _orig_regional_perf = perf_mod.regional_perf

    def _relaxed_regional_perf(*args: object, **kwargs: object) -> float:
        kwargs.setdefault("test_min_clf", 5)
        kwargs.setdefault("test_min_pos", 2)
        kwargs.setdefault("test_min_neg", 2)
        return _orig_regional_perf(*args, **kwargs)

    monkeypatch.setattr(perf_mod, "regional_perf", _relaxed_regional_perf)
    # compute_region_deltas binds regional_perf at import time via `from .perf import`.
    import transfermtl.benefits.delta as delta_mod  # noqa: E402

    monkeypatch.setattr(delta_mod, "regional_perf", _relaxed_regional_perf)

    t0 = time.time()

    # ---- 1. Train STL + pairwise MTL on the fixture ----
    train = build_synthetic_loader(synthetic_dataset, "train", ["task_1", "task_2"])
    val = build_synthetic_loader(synthetic_dataset, "val", ["task_1", "task_2"])
    test = build_synthetic_loader(synthetic_dataset, "test", ["task_1", "task_2"])

    cfg = TrainConfig(max_epochs=10, patience=10)

    stl1_pred = tmp_path / "stl1.parquet"
    stl2_pred = tmp_path / "stl2.parquet"
    mtl_pred = tmp_path / "mtl.parquet"
    mtl_ckpt = tmp_path / "mtl.pt"

    train_stl(
        train_data=[d.clone() for d in train],
        val_data=[d.clone() for d in val],
        task="task_1",
        seed=0,
        cfg=cfg,
        test_data=[d.clone() for d in test],
        predictions_path=stl1_pred,
    )
    train_stl(
        train_data=[d.clone() for d in train],
        val_data=[d.clone() for d in val],
        task="task_2",
        seed=0,
        cfg=cfg,
        test_data=[d.clone() for d in test],
        predictions_path=stl2_pred,
    )
    train_pairwise_mtl(
        train_data=[d.clone() for d in train],
        val_data=[d.clone() for d in val],
        task_i="task_1",
        task_j="task_2",
        seed=0,
        cfg=cfg,
        checkpoint_path=mtl_ckpt,
        test_data=[d.clone() for d in test],
        predictions_path=mtl_pred,
    )

    # ---- 2. Regional gradients on the MTL checkpoint per region ----
    partition = synthetic_partition
    test_smiles = {d.smi for d in test}

    g_per_region: dict[int, dict[str, np.ndarray]] = {}
    for rid in (0, 1):
        region_smis = set(partition[partition["region_id"] == rid]["smiles"])
        # Use train compounds in this region for the gradient (matches A6 semantics).
        train_in_region = [d.clone() for d in train if d.smi in region_smis]
        g1, _, _ = compute_regional_gradient(
            mtl_ckpt, task="task_1", region_data=train_in_region, task_index=0
        )
        g2, _, _ = compute_regional_gradient(
            mtl_ckpt, task="task_2", region_data=train_in_region, task_index=1
        )
        g_per_region[rid] = {"task_1": g1, "task_2": g2}

    g_a = cosine_affinity(g_per_region[0]["task_1"], g_per_region[0]["task_2"])
    g_b = cosine_affinity(g_per_region[1]["task_1"], g_per_region[1]["task_2"])
    assert g_a is not None and g_a > 0, f"G_12(A) should be positive, got {g_a}"
    assert g_b is not None and g_b < 0, f"G_12(B) should be negative, got {g_b}"

    # ---- 3. Regional benefits and S_pair ----
    # Restrict the partition to test compounds (Δ_pair is computed on test).
    test_partition = partition[partition["smiles"].isin(test_smiles)].reset_index(drop=True)
    # ---- 3a. Δ_pair end-to-end via compute_region_deltas ----
    deltas = compute_region_deltas(
        stl_preds_i=pd.read_parquet(stl1_pred),
        stl_preds_j=pd.read_parquet(stl2_pred),
        mtl_preds=pd.read_parquet(mtl_pred),
        partition=test_partition,
        task_i="task_1",
        task_j="task_2",
        task_type="clf",
    )
    assert set(deltas.keys()) == {0, 1}
    for d in deltas.values():
        assert np.isfinite(d.delta_pair), f"delta_pair NaN for region {d.region_id}"

    # ---- 3b. compute_S_pair contract on sign-heterogeneous deltas ----
    # The fixture's label-leak feature channel makes both STL and MTL fit
    # perfectly, collapsing Δ_pair to ~0. The Δ_pair *recovery* signal is
    # therefore exercised on real Tox21/SIDER data via Stage 5/7 of the pilot
    # launcher, not here. The smoke check on indices uses synthesised CIs:
    delta_results: dict[int, BootstrapResult] = {
        0: BootstrapResult(estimate=+3.0, ci_lower=+2.0, ci_upper=+4.0),
        1: BootstrapResult(estimate=-3.0, ci_lower=-4.0, ci_upper=-2.0),
    }
    s12 = compute_S_pair(delta_results, epsilon=1.5)
    assert s12, "compute_S_pair should detect sign heterogeneity on planted CIs"

    elapsed = time.time() - t0
    assert elapsed < 60.0, f"smoke pipeline took {elapsed:.1f}s (>60s)"
