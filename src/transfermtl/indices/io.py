"""Schema-validated writer for pair_indices.parquet (plan §2.9)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from transfermtl.utils.io import write_parquet
from transfermtl.utils.schemas import PairIndicesSchema

PAIR_INDICES_ROOT = Path("outputs/analysis")


def pair_indices_path(dataset: str, root: str | Path = PAIR_INDICES_ROOT) -> Path:
    return Path(root) / dataset / "pair_indices.parquet"


def write_pair_indices(
    dataset: str,
    rows: list[dict[str, object]],
    root: str | Path = PAIR_INDICES_ROOT,
) -> Path:
    """Write `rows` to pair_indices.parquet under PairIndicesSchema validation."""
    df = pd.DataFrame(rows)
    return write_parquet(pair_indices_path(dataset, root=root), df, schema=PairIndicesSchema)
