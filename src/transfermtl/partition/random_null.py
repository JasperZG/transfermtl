"""Random partitioning negative control (plan §2.6.4).

Each random partition matches the scaffold partition's region-size
distribution exactly. Procedure: shuffle compound indices, then assign the
first ``n_1`` to region 0, next ``n_2`` to region 1, etc.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from transfermtl.partition.scaffold import compute_scaffold_partition
from transfermtl.utils.io import read_parquet
from transfermtl.utils.schemas import PartitionSchema, SplitSchema


def _region_sizes(scaffold_partition: pd.DataFrame) -> list[int]:
    """Return per-region sizes ordered by region_id ascending."""
    counts = scaffold_partition["region_id"].astype(int).value_counts().sort_index()
    return [int(c) for c in counts.tolist()]


def random_partition_from_sizes(
    smiles: Sequence[str],
    region_sizes: Sequence[int],
    seed: int,
) -> pd.DataFrame:
    """Assign compounds to regions matching ``region_sizes`` exactly.

    The number of compounds must equal ``sum(region_sizes)``.
    """
    smiles_arr = np.asarray(list(smiles), dtype=object).astype(str)
    region_sizes = list(region_sizes)
    if sum(region_sizes) != len(smiles_arr):
        raise ValueError(f"sum(region_sizes)={sum(region_sizes)} != n_smiles={len(smiles_arr)}")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(smiles_arr))
    rng.shuffle(indices)

    region_id = np.empty(len(smiles_arr), dtype=int)
    cursor = 0
    for r, n in enumerate(region_sizes):
        region_id[indices[cursor : cursor + n]] = r
        cursor += n

    out = pd.DataFrame(
        {
            "smiles": smiles_arr,
            "region_id": region_id,
        }
    )
    return PartitionSchema.validate(out)


def generate_random_partitions(
    dataset: str,
    scaffold_partition: pd.DataFrame | None = None,
    n_partitions: int = 200,
    seed: int = 0,
    *,
    splits_root: str | Path = "outputs/splits",
    fps_root: str | Path = "outputs/cache/scaffold_fps",
) -> list[pd.DataFrame]:
    """Build ``n_partitions`` random-control partitions for ``dataset``.

    Each partition's region-size distribution exactly matches the primary
    scaffold partition's. The b-th partition uses ``rng_seed = seed + b``.
    """
    split_path = Path(splits_root) / dataset / "split.parquet"
    df_split = read_parquet(split_path, schema=SplitSchema)
    smiles_list = df_split["smiles"].astype(str).tolist()

    if scaffold_partition is None:
        scaffold_partition = compute_scaffold_partition(
            dataset, splits_root=splits_root, fps_root=fps_root
        )
    scaffold_partition = PartitionSchema.validate(scaffold_partition)
    region_sizes = _region_sizes(scaffold_partition)

    return [
        random_partition_from_sizes(smiles_list, region_sizes, seed=seed + b)
        for b in range(n_partitions)
    ]
