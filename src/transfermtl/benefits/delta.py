"""Regional transfer benefits (plan §2.8).

For each region r and pair (i, j), compute four quantities from saved STL +
pairwise-MTL prediction parquets:

    Δ_{i ← j}(r) = Perf_MTL_i(r) − Perf_STL_i(r)
    Δ_{j ← i}(r) = Perf_MTL_j(r) − Perf_STL_j(r)
    Δ_ij(r)      = (Δ_{i ← j} + Δ_{j ← i}) / 2
    Δ_ij^worst(r) = min(Δ_{i ← j}, Δ_{j ← i})

A unit AUC point is small (~0.01); the paper's ε threshold for sign
heterogeneity is 1.5 *AUC points* (plan §2.21), i.e. Δ values are reported
in points. Callers should multiply ROC-AUC differences by 100 if they want
to use ε = 1.5 directly. We report Δ in raw AUC units; the indices module
applies the scaling (default ε = 1.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from transfermtl.benefits.perf import TaskType, regional_perf


@dataclass(frozen=True)
class RegionDeltas:
    region_id: int
    delta_pair: float
    delta_i_from_j: float
    delta_j_from_i: float
    delta_worst: float
    n_test: int


def _region_to_compounds(partition: pd.DataFrame) -> dict[int, set[str]]:
    return {
        int(rid): set(grp["smiles"].astype(str).tolist())
        for rid, grp in partition.groupby("region_id")
    }


def compute_region_deltas(
    stl_preds_i: pd.DataFrame,
    stl_preds_j: pd.DataFrame,
    mtl_preds: pd.DataFrame,
    partition: pd.DataFrame,
    task_i: str,
    task_j: str,
    task_type: TaskType = "clf",
    test_min_clf: int | None = None,
    test_min_reg: int | None = None,
    test_min_pos: int | None = None,
    test_min_neg: int | None = None,
) -> dict[int, RegionDeltas]:
    """Per-region Δ values from prediction parquets + a partition table.

    All prediction frames must satisfy `PredictionSchema` (smiles, task, y_true,
    y_pred, seed). The MTL frame is expected to contain rows for both tasks.

    Threshold kwargs default to None and forward only when set, so a caller
    that monkey-patches `regional_perf` (e.g. relaxed validity for tests) is
    not silently overridden by this function's defaults.
    """
    out: dict[int, RegionDeltas] = {}
    perf_kwargs: dict[str, int] = {}
    if test_min_clf is not None:
        perf_kwargs["test_min_clf"] = test_min_clf
    if test_min_reg is not None:
        perf_kwargs["test_min_reg"] = test_min_reg
    if test_min_pos is not None:
        perf_kwargs["test_min_pos"] = test_min_pos
    if test_min_neg is not None:
        perf_kwargs["test_min_neg"] = test_min_neg

    for region_id, smis in _region_to_compounds(partition).items():
        compounds = list(smis)
        n_test = sum(
            1
            for s in compounds
            if s
            in set(mtl_preds["smiles"]) | set(stl_preds_i["smiles"]) | set(stl_preds_j["smiles"])
        )

        stl_i = regional_perf(stl_preds_i, compounds, task_i, task_type=task_type, **perf_kwargs)
        stl_j = regional_perf(stl_preds_j, compounds, task_j, task_type=task_type, **perf_kwargs)
        mtl_i = regional_perf(mtl_preds, compounds, task_i, task_type=task_type, **perf_kwargs)
        mtl_j = regional_perf(mtl_preds, compounds, task_j, task_type=task_type, **perf_kwargs)

        d_i = mtl_i - stl_i
        d_j = mtl_j - stl_j
        d_pair = 0.5 * (d_i + d_j)
        d_worst = min(d_i, d_j)
        out[region_id] = RegionDeltas(
            region_id=region_id,
            delta_pair=float(d_pair),
            delta_i_from_j=float(d_i),
            delta_j_from_i=float(d_j),
            delta_worst=float(d_worst),
            n_test=int(n_test),
        )

    return out
