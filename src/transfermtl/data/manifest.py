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


def load_task_rename(
    dataset: str,
    manifest_dir: Path | None = None,
) -> dict[str, str]:
    """Return the friendly→column-name mapping written by `prepare_dataset`.

    A2's data pipeline stores original task labels (e.g. ``NR-AR``) under
    canonical ``task_1..task_K`` columns to keep ``SplitSchema`` clean and
    avoid filesystem-unfriendly characters in the parquet schema. The
    original→canonical map is persisted under the ``task_rename`` key in
    ``outputs/data_manifest/{dataset}.json``.

    Returns an empty dict when the manifest is absent (e.g. the synthetic
    test fixture, where columns are already named ``task_1`` / ``task_2``).
    """
    mdir = manifest_dir or MANIFEST_DIR
    json_path = mdir / f"{dataset}.json"
    if not json_path.exists():
        return {}
    sidecar = json.loads(json_path.read_text())
    rename = sidecar.get("task_rename")
    if not isinstance(rename, dict):
        return {}
    return {str(k): str(v) for k, v in rename.items()}


def resolve_task_name(
    name: str,
    dataset: str,
    available_columns: list[str] | None = None,
    manifest_dir: Path | None = None,
) -> str:
    """Resolve a CLI-provided task name to its split-parquet column name.

    - If ``name`` is already present in ``available_columns``, return it
      unchanged (already a canonical column).
    - Otherwise look up ``name`` in the manifest's ``task_rename`` map and
      return the mapped column name.
    - If neither resolves, return ``name`` as-is so the caller can raise a
      clear ``KeyError`` against the parquet.
    """
    if available_columns is not None and name in available_columns:
        return name
    rename = load_task_rename(dataset, manifest_dir=manifest_dir)
    if name in rename:
        return rename[name]
    return name
