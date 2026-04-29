"""Dataset registry: Tox21 + SIDER loaders, plus prepare_dataset entry point.

Loaders pull from MoleculeNet's canonical CSVs (mirrored on the DeepChem S3
bucket); a local copy under data/raw/{dataset}/ is preferred when present so
offline / air-gapped reruns work.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from transfermtl.data.featurize import featurize_and_cache
from transfermtl.data.fingerprints import cache_scaffold_fingerprints
from transfermtl.data.manifest import MANIFEST_DIR, write_manifest
from transfermtl.data.scaffolds import compute_scaffold
from transfermtl.data.splits import scaffold_stratified_split
from transfermtl.data.standardize import standardize_smiles
from transfermtl.utils.config import SHARED_DIR
from transfermtl.utils.registry import register

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
SIDER_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"

TOX21_TASKS = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, attempts: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read()
            dest.write_bytes(raw)
            return dest
        except Exception as exc:
            last_err = exc
            log.warning("download attempt %d for %s failed: %s", i + 1, url, exc)
            time.sleep(2**i)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def _read_csv_gz_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


@register("dataset", "tox21")
def load_tox21() -> pd.DataFrame:
    """Return Tox21 with columns: smiles + 12 task columns. Labels in {0, 1, NaN}."""
    raw = RAW_DIR / "tox21" / "tox21.csv.gz"
    _download(TOX21_URL, raw)
    df = _read_csv_gz_or_csv(raw)
    cols = ["smiles", *TOX21_TASKS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Tox21 CSV missing columns: {missing}")
    return df[cols].copy()


@register("dataset", "sider")
def load_sider() -> pd.DataFrame:
    """Return SIDER with columns: smiles + 27 side-effect task columns."""
    raw = RAW_DIR / "sider" / "sider.csv.gz"
    _download(SIDER_URL, raw)
    df = _read_csv_gz_or_csv(raw)
    if "smiles" not in df.columns:
        raise ValueError("SIDER CSV missing 'smiles' column")
    task_cols = [c for c in df.columns if c != "smiles"]
    return df[["smiles", *task_cols]].copy()


_RESERVED_COLS = ("smiles", "scaffold", "split")


def _rename_to_task_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename non-reserved columns to task_1..task_K (SplitSchema regex).

    Reserved columns (smiles, scaffold, split) are preserved as-is.
    """
    label_cols = [c for c in df.columns if c not in _RESERVED_COLS]
    rename = {c: f"task_{i + 1}" for i, c in enumerate(label_cols)}
    out = df.rename(columns=rename)
    return out, rename


def prepare_dataset(
    name: str,
    force: bool = False,
    do_featurize: bool = True,
) -> Path:
    """Load + standardize + scaffold + featurize-cache + split + write manifest.

    Idempotent: returns early if outputs already exist with a manifest matching
    the current raw-input hash.
    """
    from transfermtl.utils.config import load_config

    # Touch the lock file via load_config so we fail fast on hyperparameter drift.
    load_config(SHARED_DIR / "preprocess.yaml")

    if name == "tox21":
        df = load_tox21()
        raw_path = RAW_DIR / "tox21" / "tox21.csv.gz"
    elif name == "sider":
        df = load_sider()
        raw_path = RAW_DIR / "sider" / "sider.csv.gz"
    else:
        raise KeyError(f"unknown dataset {name!r}; only tox21 and sider are wired in this wave")

    raw_hash = _file_sha256(raw_path)

    out_parquet = Path("outputs/splits") / name / "split.parquet"
    sidecar = MANIFEST_DIR / f"{name}.json"
    if not force and out_parquet.exists() and sidecar.exists():
        previous = json.loads(sidecar.read_text())
        if previous.get("raw_inputs_sha256", {}).get(raw_path.name) == raw_hash:
            log.info("skip: %s already prepared (matching raw hash)", name)
            return out_parquet

    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    canonical = [standardize_smiles(s) for s in df["smiles"].tolist()]
    df["canonical_smiles"] = canonical
    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    df["smiles"] = df["canonical_smiles"]
    df = df.drop(columns=["canonical_smiles"])

    df = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    df["scaffold"] = [compute_scaffold(s) for s in df["smiles"]]

    cache_scaffold_fingerprints(name, df["scaffold"].tolist(), force=force)

    if do_featurize:
        for s in df["smiles"]:
            featurize_and_cache(s)

    df["split"] = scaffold_stratified_split(df, seed=42)

    df, _renames = _rename_to_task_columns(df)
    task_cols = [c for c in df.columns if c.startswith("task_")]
    for c in task_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    df = df[["smiles", "scaffold", "split", *task_cols]]

    parquet_path, _ = write_manifest(
        name,
        df,
        raw_inputs={raw_path.name: raw_hash},
        extra={"task_rename": _renames},
    )
    return parquet_path


def list_registered_datasets() -> Iterable[str]:
    from transfermtl.utils.registry import list_registered

    return list_registered("dataset")
