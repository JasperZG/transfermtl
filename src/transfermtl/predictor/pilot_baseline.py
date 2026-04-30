"""Pilot-stage predictor for the §5.7 predictor green-light check.

This is intentionally minimal: it computes one feature (``G_ij(r)`` from A6)
and three baselines (scaffold tanimoto, embedding distance, label correlation),
each trying to predict ``sign(Δ_ij(r))`` across the full pool of (pair, region,
seed) rows surviving the §2.11 / §2.12 validity filters. A8 generalises this
into the full Phase 2 predictor and evaluates incremental R² (plan §2.14); A7's
pilot baseline is only here so the gate's predictor sub-criteria can be scored.

The functions below all operate on in-memory pandas frames so they can be
exercised in unit tests with synthetic inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class PredictorScores:
    """All scalars the §5.7 predictor sub-criteria require.

    `auroc_baselines` is a {baseline_name: auroc} mapping; the strongest is
    promoted into `auroc_best_baseline` / `best_baseline_name`.
    `cross_seed_agreement` is the fraction of (pair, region) cells where
    ``sign(G_ij(r))`` is identical across **all** seeds (NaN cells excluded).
    """

    n_rows: int
    auroc_g_ij: float
    auroc_baselines: dict[str, float]
    auroc_best_baseline: float
    best_baseline_name: str
    spearman_rho: float
    cross_seed_agreement: float


def _safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC with NaN-row dropping; returns NaN when undefined."""
    mask = ~(np.isnan(y_true) | np.isnan(score))
    if mask.sum() < 2:
        return float("nan")
    yt = y_true[mask].astype(int)
    if yt.min() == yt.max():
        return float("nan")
    return float(roc_auc_score(yt, score[mask]))


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    rho = spearmanr(a[mask], b[mask]).correlation
    return float(rho) if rho == rho else float("nan")


def cross_seed_sign_agreement(
    rows: pd.DataFrame,
    *,
    g_col: str = "G_ij",
    seed_col: str = "seed",
    pair_col: str = "pair_id",
    region_col: str = "region_id",
) -> float:
    """Return the fraction of (pair, region) cells where sign(G_ij) is unanimous
    across every seed for that cell.

    Cells where the single-seed estimate is NaN or zero are excluded from the
    cell as missing; cells with fewer than 2 seeds after filtering are dropped.
    """
    df = rows[[pair_col, region_col, seed_col, g_col]].copy()
    df = df.dropna(subset=[g_col])
    df = df[df[g_col] != 0.0]
    if df.empty:
        return float("nan")
    df["sign"] = np.sign(df[g_col]).astype(int)
    grouped = df.groupby([pair_col, region_col])["sign"].agg(n_seeds="count", n_unique="nunique")
    grouped = grouped[grouped["n_seeds"] >= 2]
    if grouped.empty:
        return float("nan")
    return float((grouped["n_unique"] == 1).mean())


def evaluate_pilot_predictor(
    measurements: pd.DataFrame,
    *,
    g_col: str = "G_ij",
    delta_col: str = "delta_pair",
    baseline_cols: Sequence[str] = (
        "scaffold_tanimoto",
        "embedding_distance",
        "label_correlation",
    ),
    seed_col: str = "seed",
    pair_col: str = "pair_id",
    region_col: str = "region_id",
) -> PredictorScores:
    """Compute the four §5.7 predictor stats on a flat measurements frame.

    Required columns: ``g_col``, ``delta_col``, every entry in ``baseline_cols``,
    ``seed_col``, ``pair_col``, ``region_col``. NaN rows are dropped per metric.
    """
    df = measurements
    y = np.sign(df[delta_col].to_numpy())
    y = np.where(y == 0.0, np.nan, y)
    y_true = (y > 0).astype(float)
    y_true[np.isnan(y)] = float("nan")

    auroc_g = _safe_auroc(y_true, df[g_col].to_numpy())

    auroc_baselines: dict[str, float] = {}
    for col in baseline_cols:
        if col not in df.columns:
            auroc_baselines[col] = float("nan")
            continue
        auroc_baselines[col] = _safe_auroc(y_true, df[col].to_numpy())

    finite_baselines = {k: v for k, v in auroc_baselines.items() if not np.isnan(v)}
    if finite_baselines:
        best_name = max(finite_baselines, key=lambda k: finite_baselines[k])
        best_value = finite_baselines[best_name]
    else:
        best_name = ""
        best_value = float("nan")

    rho = _safe_spearman(df[g_col].to_numpy(), df[delta_col].to_numpy())

    agreement = cross_seed_sign_agreement(
        df, g_col=g_col, seed_col=seed_col, pair_col=pair_col, region_col=region_col
    )

    return PredictorScores(
        n_rows=int(len(df)),
        auroc_g_ij=auroc_g,
        auroc_baselines=auroc_baselines,
        auroc_best_baseline=best_value,
        best_baseline_name=best_name,
        spearman_rho=rho,
        cross_seed_agreement=agreement,
    )


# ---------------------------------------------------------------------------
# Inline baseline computations used by the pilot orchestration script.
#
# Each helper returns a per-(pair, region, seed) score column so the caller
# can join it onto the measurements frame. A8 will generalise these into a
# shared feature module, so we keep the surface deliberately small here.
# ---------------------------------------------------------------------------


def scaffold_tanimoto_baseline(
    region_compounds: Mapping[tuple[str, int, str], list[int]],
    fingerprints: Mapping[str, np.ndarray],
    scaffolds: Mapping[str, str],
    smiles_lookup: Mapping[str, list[str]],
) -> dict[tuple[str, int, str], float]:
    """Mean Tanimoto distance between scaffolds in D_i(r) and D_j(r).

    `region_compounds[(pair_id, region_id, side)]` is the list of compound
    indices contributing to side ``"i"`` or ``"j"`` in that region. ``smiles_lookup``
    is keyed by ``pair_id`` and lists the dataset's compounds in row order.
    """
    out: dict[tuple[str, int, str], float] = {}
    for (pair_id, region_id, side), idxs in region_compounds.items():
        if side != "i":
            continue
        j_key = (pair_id, region_id, "j")
        if j_key not in region_compounds:
            out[(pair_id, region_id, "i_vs_j")] = float("nan")
            continue
        smis = smiles_lookup[pair_id]
        scaff_i = {scaffolds[smis[k]] for k in idxs}
        scaff_j = {scaffolds[smis[k]] for k in region_compounds[j_key]}
        if not scaff_i or not scaff_j:
            out[(pair_id, region_id, "i_vs_j")] = float("nan")
            continue
        sims: list[float] = []
        for sa in scaff_i:
            fa = fingerprints.get(sa)
            if fa is None:
                continue
            for sb in scaff_j:
                fb = fingerprints.get(sb)
                if fb is None:
                    continue
                inter = float(np.bitwise_and(fa, fb).sum())
                union = float(np.bitwise_or(fa, fb).sum())
                if union == 0:
                    sims.append(0.0)
                else:
                    sims.append(inter / union)
        out[(pair_id, region_id, "i_vs_j")] = float(np.mean(sims)) if sims else float("nan")
    return out


def embedding_distance_baseline(
    region_embeddings_i: Mapping[tuple[str, int], np.ndarray],
    region_embeddings_j: Mapping[tuple[str, int], np.ndarray],
) -> dict[tuple[str, int], float]:
    """L2 distance between mean encoder embeddings of D_i(r) and D_j(r)."""
    out: dict[tuple[str, int], float] = {}
    for key, ei in region_embeddings_i.items():
        ej = region_embeddings_j.get(key)
        if ej is None or ei.size == 0 or ej.size == 0:
            out[key] = float("nan")
            continue
        out[key] = float(np.linalg.norm(ei - ej))
    return out


def label_correlation_baseline(
    region_labels: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[str, int], float]:
    """Pearson correlation between co-measured labels for tasks i and j in a region.

    `region_labels[(pair_id, region_id)] = (y_i, y_j)` where each array contains
    only co-measured compounds (no NaNs). Returns NaN when fewer than 5 pairs.
    """
    out: dict[tuple[str, int], float] = {}
    for key, (yi, yj) in region_labels.items():
        if yi.size < 5 or yj.size < 5 or yi.size != yj.size:
            out[key] = float("nan")
            continue
        if yi.std() == 0.0 or yj.std() == 0.0:
            out[key] = float("nan")
            continue
        out[key] = float(np.corrcoef(yi, yj)[0, 1])
    return out
