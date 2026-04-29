"""Partition writer — schema-validated parquet files under ``outputs/partitions/``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from transfermtl.utils.io import write_parquet
from transfermtl.utils.schemas import PartitionSchema

Scheme = Literal["scaffold", "latent", "knn", "random"]
PARTITIONS_ROOT = Path("outputs/partitions")


def partition_path(
    dataset: str,
    scheme: Scheme,
    M: int | None = None,
    b: int | None = None,
    root: str | Path = PARTITIONS_ROOT,
) -> Path:
    """Canonical filesystem path for a partition parquet.

    - ``scaffold|latent|knn`` → ``{root}/{dataset}/{scheme}_M{M}.parquet``.
    - ``random`` → ``{root}/{dataset}/random_b{b}.parquet``.
    """
    root_p = Path(root)
    if scheme == "random":
        if b is None:
            raise ValueError("scheme='random' requires b")
        return root_p / dataset / f"random_b{b}.parquet"
    if M is None:
        raise ValueError(f"scheme='{scheme}' requires M")
    return root_p / dataset / f"{scheme}_M{M}.parquet"


def write_partition(
    dataset: str,
    scheme: Scheme,
    df: pd.DataFrame,
    M: int | None = None,
    b: int | None = None,
    root: str | Path = PARTITIONS_ROOT,
) -> Path:
    """Validate against PartitionSchema and write to canonical path."""
    path = partition_path(dataset, scheme, M=M, b=b, root=root)
    return write_parquet(path, df, schema=PartitionSchema)
