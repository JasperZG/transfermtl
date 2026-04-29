"""IO for meaningful-pair results (plan §2.12).

Writes outputs/analysis/{dataset}/meaningful_pairs.parquet, validated against
MeaningfulPairSchema. Schema: pair_id (str), is_meaningful (bool),
failed_reasons (list[str], stored as object).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from transfermtl.utils.io import write_parquet
from transfermtl.utils.schemas import MeaningfulPairSchema

OUTPUTS_DIR = Path("outputs")


def meaningful_pairs_path(dataset: str, base: Path | None = None) -> Path:
    root = base if base is not None else OUTPUTS_DIR
    return root / "analysis" / dataset / "meaningful_pairs.parquet"


def write_meaningful_pairs(
    dataset: str,
    results: Iterable[tuple[str, bool, list[str]]],
    base: Path | None = None,
) -> Path:
    """Write meaningful-pair decisions to a schema-valid parquet.

    `results` is an iterable of `(pair_id, is_meaningful, failed_reasons)`.
    """
    rows = [
        {
            "pair_id": pair_id,
            "is_meaningful": bool(is_meaningful),
            "failed_reasons": list(failed_reasons),
        }
        for pair_id, is_meaningful, failed_reasons in results
    ]
    df = pd.DataFrame(rows, columns=["pair_id", "is_meaningful", "failed_reasons"])
    return write_parquet(meaningful_pairs_path(dataset, base), df, schema=MeaningfulPairSchema)
