"""Per-dataset manifest writer.

Emits the canonical split.parquet (validated against SplitSchema) and a JSON
sidecar capturing the inputs and library versions used. The hash of the raw
input file is included so prepare_dataset.py can implement idempotent reruns.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from transfermtl.utils.git import current_sha
from transfermtl.utils.io import write_parquet
from transfermtl.utils.schemas import SplitSchema

SPLITS_DIR = Path("outputs/splits")
MANIFEST_DIR = Path("outputs/data_manifest")


def write_manifest(
    dataset: str,
    df: pd.DataFrame,
    raw_inputs: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write split.parquet and the JSON sidecar for `dataset`.

    Args:
        dataset: dataset key (e.g. "tox21").
        df: dataframe with columns `smiles, scaffold, split, task_*` matching
            SplitSchema. Must already contain the split assignment column.
        raw_inputs: map of input-file label -> sha256 of that file.
        extra: optional fields to include in the JSON (e.g. RDKit version).

    Returns (parquet_path, json_path).
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = SPLITS_DIR / dataset / "split.parquet"
    write_parquet(parquet_path, df, schema=SplitSchema)

    task_cols = [c for c in df.columns if c.startswith("task_")]
    per_split = df["split"].value_counts().to_dict()
    per_task_n = {col: int(df[col].notna().sum()) for col in task_cols}

    sidecar: dict[str, Any] = {
        "dataset": dataset,
        "row_count": int(len(df)),
        "scaffold_count": int(df["scaffold"].nunique()),
        "per_split_counts": {k: int(v) for k, v in per_split.items()},
        "task_columns": task_cols,
        "per_task_label_count": per_task_n,
        "raw_inputs_sha256": raw_inputs,
        "git_sha": current_sha(),
        "written_at": datetime.now(UTC).isoformat(),
        "library_versions": _library_versions(),
    }
    if extra:
        sidecar.update(extra)

    json_path = MANIFEST_DIR / f"{dataset}.json"
    json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return parquet_path, json_path


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("rdkit", "pandas", "numpy", "torch", "torch_geometric"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "missing"
    return versions
