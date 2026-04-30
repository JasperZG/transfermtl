"""Schema-validated writer for region_affinity.parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from transfermtl.utils.io import write_parquet
from transfermtl.utils.schemas import GradientAffinitySchema

GRADIENTS_ROOT = Path("outputs/gradients")


def region_affinity_path(
    dataset: str,
    task_i: str,
    task_j: str,
    seed: int,
    root: str | Path = GRADIENTS_ROOT,
) -> Path:
    return Path(root) / dataset / f"{task_i}_{task_j}" / f"seed{seed}" / "region_affinity.parquet"


def write_region_affinity(
    dataset: str,
    task_i: str,
    task_j: str,
    seed: int,
    rows: list[dict[str, object]],
    root: str | Path = GRADIENTS_ROOT,
) -> Path:
    """Validate `rows` against GradientAffinitySchema and persist.

    `rows` keys must match the schema columns: region_id, G_ij, g_i_norm,
    g_j_norm, n_i_in_region, n_j_in_region, checkpoint_label, seed.
    """
    df = pd.DataFrame(rows)
    path = region_affinity_path(dataset, task_i, task_j, seed, root=root)
    return write_parquet(path, df, schema=GradientAffinitySchema)
