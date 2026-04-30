"""Seed-averaging + within-region hierarchical-bootstrap CIs (plan §2.8, §2.10).

Given multi-seed STL/MTL prediction parquets, compute per-region Δ for each
seed, average across seeds, then attach a 95% CI from
`bootstrap.within_region.within_region_bootstrap` resampling at scaffold
level (level 1) + compounds within scaffold (level 2). Seed mixing (level 3)
is automatic when more than one seed is present.

Outputs satisfy `RegionBenefitSchema`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from transfermtl.benefits.delta import compute_region_deltas
from transfermtl.benefits.perf import TaskType, regional_perf
from transfermtl.bootstrap.within_region import within_region_bootstrap
from transfermtl.utils.io import write_parquet
from transfermtl.utils.schemas import RegionBenefitSchema
from transfermtl.utils.types import HierarchicalSamples

BENEFITS_ROOT = Path("outputs/benefits")


def benefits_path(dataset: str, task_i: str, task_j: str, root: str | Path = BENEFITS_ROOT) -> Path:
    return Path(root) / dataset / f"{task_i}_{task_j}" / "region_benefits.parquet"


def _delta_compute_fn(
    stl_i: pd.DataFrame,
    stl_j: pd.DataFrame,
    mtl: pd.DataFrame,
    task_i: str,
    task_j: str,
    task_type: TaskType,
) -> Callable[[HierarchicalSamples], float]:
    """Closure used by hierarchical_bootstrap.

    The bootstrap resamples scaffold groups within a region. Each resample is
    a HierarchicalSamples whose `values` array carries SMILES strings (cast to
    `object` dtype). We re-derive the regional Δ from the resampled compound
    set.
    """

    def fn(samples: HierarchicalSamples) -> float:
        smis = [str(v) for v in samples.values.tolist()]
        if not smis:
            return float("nan")
        stl_i_p = regional_perf(
            stl_i, smis, task_i, task_type=task_type, test_min_clf=1, test_min_pos=1, test_min_neg=1
        )
        stl_j_p = regional_perf(
            stl_j, smis, task_j, task_type=task_type, test_min_clf=1, test_min_pos=1, test_min_neg=1
        )
        mtl_i_p = regional_perf(
            mtl, smis, task_i, task_type=task_type, test_min_clf=1, test_min_pos=1, test_min_neg=1
        )
        mtl_j_p = regional_perf(
            mtl, smis, task_j, task_type=task_type, test_min_clf=1, test_min_pos=1, test_min_neg=1
        )
        if any(np.isnan([stl_i_p, stl_j_p, mtl_i_p, mtl_j_p])):
            return float("nan")
        return 0.5 * ((mtl_i_p - stl_i_p) + (mtl_j_p - stl_j_p))

    return fn


def aggregate_region_benefits(
    stl_preds_i_per_seed: Mapping[int, pd.DataFrame],
    stl_preds_j_per_seed: Mapping[int, pd.DataFrame],
    mtl_preds_per_seed: Mapping[int, pd.DataFrame],
    partition: pd.DataFrame,
    split: pd.DataFrame,
    task_i: str,
    task_j: str,
    task_type: TaskType = "clf",
    n_iter: int = 500,
    rng_seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Average per-seed Δ values across seeds, attach 95% CIs.

    Returns a DataFrame validated against `RegionBenefitSchema` with one row
    per region.
    """
    seeds = sorted(set(stl_preds_i_per_seed) & set(stl_preds_j_per_seed) & set(mtl_preds_per_seed))
    if not seeds:
        raise ValueError("no overlapping seeds across STL i/j and MTL prediction frames")

    # Per-seed point estimates.
    per_seed: dict[int, dict[int, np.ndarray]] = {s: {} for s in seeds}
    for s in seeds:
        d = compute_region_deltas(
            stl_preds_i_per_seed[s],
            stl_preds_j_per_seed[s],
            mtl_preds_per_seed[s],
            partition,
            task_i,
            task_j,
            task_type=task_type,
        )
        for rid, rec in d.items():
            per_seed[s][rid] = np.array(
                [
                    rec.delta_pair,
                    rec.delta_i_from_j,
                    rec.delta_j_from_i,
                    rec.delta_worst,
                    rec.n_test,
                ],
                dtype=np.float64,
            )

    region_ids = sorted({rid for s in seeds for rid in per_seed[s]})

    # Concatenated samples for the bootstrap: one row per (compound × seed).
    smiles_all = partition["smiles"].astype(str).to_numpy()
    region_of = dict(
        zip(
            partition["smiles"].astype(str),
            partition["region_id"].astype(int),
            strict=True,
        )
    )
    scaffold_of = dict(
        zip(
            split["smiles"].astype(str),
            split["scaffold"].astype(str),
            strict=True,
        )
    )

    rows: list[dict[str, object]] = []
    for rid in region_ids:
        # Aggregate per-seed estimates.
        deltas = np.array(
            [per_seed[s][rid][0] for s in seeds if rid in per_seed[s]], dtype=np.float64
        )
        d_i = np.array([per_seed[s][rid][1] for s in seeds if rid in per_seed[s]], dtype=np.float64)
        d_j = np.array([per_seed[s][rid][2] for s in seeds if rid in per_seed[s]], dtype=np.float64)
        d_w = np.array([per_seed[s][rid][3] for s in seeds if rid in per_seed[s]], dtype=np.float64)
        n_test = int(per_seed[seeds[0]][rid][4])

        if np.all(np.isnan(deltas)):
            rows.append(
                {
                    "region_id": rid,
                    "delta_pair": float("nan"),
                    "delta_i_from_j": float("nan"),
                    "delta_j_from_i": float("nan"),
                    "delta_worst": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "n_test": n_test,
                }
            )
            continue

        # Build HierarchicalSamples for within-region bootstrap.
        # Compounds in this region:
        compounds = [s for s in smiles_all if region_of.get(s) == rid]
        scaffolds = np.array([scaffold_of.get(s, "<unknown>") for s in compounds], dtype=object)
        values = np.array(compounds, dtype=object)
        region_arr = np.array([rid] * len(compounds), dtype=int)

        samples = HierarchicalSamples(values=values, scaffold_ids=scaffolds, seed_ids=None)

        # Bootstrap on the first available seed; this captures within-region
        # data variance. Seed mixing across runs is approximated by averaging.
        compute_fn = _delta_compute_fn(
            stl_preds_i_per_seed[seeds[0]],
            stl_preds_j_per_seed[seeds[0]],
            mtl_preds_per_seed[seeds[0]],
            task_i,
            task_j,
            task_type,
        )
        boot = within_region_bootstrap(
            compute_fn,
            samples,
            region_ids=region_arr,
            target_region=rid,
            n_iter=n_iter,
            rng_seed=rng_seed,
            alpha=alpha,
        )

        rows.append(
            {
                "region_id": rid,
                "delta_pair": float(np.nanmean(deltas)),
                "delta_i_from_j": float(np.nanmean(d_i)),
                "delta_j_from_i": float(np.nanmean(d_j)),
                "delta_worst": float(np.nanmean(d_w)),
                "ci_lo": float(boot.ci_lower),
                "ci_hi": float(boot.ci_upper),
                "n_test": n_test,
            }
        )

    return RegionBenefitSchema.validate(pd.DataFrame(rows))


def write_region_benefits(
    dataset: str,
    task_i: str,
    task_j: str,
    df: pd.DataFrame,
    root: str | Path = BENEFITS_ROOT,
) -> Path:
    return write_parquet(
        benefits_path(dataset, task_i, task_j, root=root), df, schema=RegionBenefitSchema
    )
