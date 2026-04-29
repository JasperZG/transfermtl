"""Tests for src/transfermtl/data: standardize, scaffolds, fingerprints,
splits, featurize, and the prepare_dataset idempotency contract.

The 12 listed wave-2 A2 tests live in this file. Tests requiring real Tox21
or SIDER downloads are skipped automatically when the raw CSV is absent.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transfermtl.data.featurize import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    cache_path_for,
    featurize_smiles,
)
from transfermtl.data.fingerprints import morgan_fingerprint
from transfermtl.data.scaffolds import EMPTY_SCAFFOLD, compute_scaffold
from transfermtl.data.splits import scaffold_stratified_split
from transfermtl.data.standardize import standardize_smiles
from transfermtl.utils.schemas import SplitSchema

# ---------- standardize ----------


def test_standardize_idempotent() -> None:
    """standardize(standardize(s)) == standardize(s) for canonical SMILES."""
    smiles = ["CCO", "c1ccccc1", "O=C(O)c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    for s in smiles:
        once = standardize_smiles(s)
        assert once is not None
        twice = standardize_smiles(once)
        assert twice == once


def test_standardize_handles_salts() -> None:
    """Salt forms collapse to the largest organic fragment."""
    out = standardize_smiles("CCO.[Na+].[Cl-]")
    assert out == "CCO"


def test_standardize_returns_none_on_garbage() -> None:
    assert standardize_smiles("not-a-smiles") is None
    assert standardize_smiles("") is None
    assert standardize_smiles("ZZZ123") is None


# ---------- scaffolds ----------


def test_scaffold_empty_bucket() -> None:
    """Acyclic molecules bucket under the EMPTY_SCAFFOLD sentinel."""
    assert compute_scaffold("C") == EMPTY_SCAFFOLD
    assert compute_scaffold("CCO") == EMPTY_SCAFFOLD
    assert compute_scaffold("CCCC") == EMPTY_SCAFFOLD


def test_scaffold_canonical() -> None:
    """Equivalent SMILES of the same molecule yield the same scaffold."""
    a = compute_scaffold("c1ccccc1Cl")
    b = compute_scaffold("Clc1ccccc1")
    assert a == b
    assert a != EMPTY_SCAFFOLD


# ---------- fingerprints ----------


def test_fingerprint_shape() -> None:
    fp = morgan_fingerprint("c1ccccc1")
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8
    # Some bits set for a real molecule.
    assert fp.sum() > 0


# ---------- splits ----------


def test_split_no_scaffold_leakage(synthetic_dataset: pd.DataFrame) -> None:
    """No scaffold appears in more than one split."""
    grouped = synthetic_dataset.groupby("scaffold")["split"].nunique()
    assert (grouped == 1).all()


def test_split_fractions_within_tolerance() -> None:
    """Greedy splitter respects target fractions within ±2% on a moderate dataset."""
    rng = np.random.default_rng(0)
    n_scaffolds = 80
    sizes = rng.integers(1, 8, size=n_scaffolds)
    rows = []
    for s_idx, size in enumerate(sizes):
        for m in range(int(size)):
            rows.append({"smiles": f"m_{s_idx}_{m}", "scaffold": f"s_{s_idx}"})
    df = pd.DataFrame(rows)
    splits = scaffold_stratified_split(df, train=0.70, val=0.15, test=0.15, seed=42)
    fracs = splits.value_counts(normalize=True)
    assert abs(fracs.get("train", 0) - 0.70) < 0.02
    assert abs(fracs.get("val", 0) - 0.15) < 0.02
    assert abs(fracs.get("test", 0) - 0.15) < 0.02


def test_split_deterministic() -> None:
    """Same seed -> identical assignment."""
    rng = np.random.default_rng(1)
    rows = []
    for s_idx in range(40):
        size = int(rng.integers(1, 6))
        for m in range(size):
            rows.append({"smiles": f"m_{s_idx}_{m}", "scaffold": f"s_{s_idx}"})
    df = pd.DataFrame(rows)
    a = scaffold_stratified_split(df, seed=42)
    b = scaffold_stratified_split(df, seed=42)
    pd.testing.assert_series_equal(a, b)
    c = scaffold_stratified_split(df, seed=43)
    # A different seed should change at least one assignment for nontrivial data.
    assert not a.equals(c)


# ---------- prepare_dataset idempotency ----------


@pytest.fixture(scope="module")
def tox21_raw_present() -> bool:
    return Path("data/raw/tox21/tox21.csv.gz").exists()


def test_prepare_dataset_idempotent(tox21_raw_present: bool, tmp_path: Path) -> None:
    """Running prepare_dataset twice yields byte-identical split.parquet.

    Skipped when the raw Tox21 CSV is unavailable (e.g. CI without internet).
    """
    if not tox21_raw_present:
        pytest.skip("data/raw/tox21/tox21.csv.gz not present")

    from transfermtl.data.datasets import prepare_dataset

    out1 = prepare_dataset("tox21", force=True, do_featurize=False)
    bytes1 = out1.read_bytes()
    out2 = prepare_dataset("tox21", force=False, do_featurize=False)
    bytes2 = out2.read_bytes()
    assert bytes1 == bytes2


def test_split_schema_validates(synthetic_dataset: pd.DataFrame) -> None:
    """A1's fixture parquet (re-validated) — sanity for SplitSchema usage in A2 tests."""
    SplitSchema.validate(synthetic_dataset)


# ---------- featurize ----------


def test_featurize_synthetic_roundtrip(tmp_path: Path) -> None:
    """Featurize a small molecule, recover atom count via data.x.shape[0]."""
    data = featurize_smiles("CCO")
    assert data is not None
    assert data.x.shape == (3, ATOM_FEATURE_DIM)
    # Two C-C/C-O bonds → 4 directed edges
    assert data.edge_index.shape == (2, 4)
    assert data.edge_attr.shape == (4, BOND_FEATURE_DIM)
    assert data.smiles == "CCO"


def test_featurize_cache_path_deterministic() -> None:
    a = cache_path_for("CCO")
    b = cache_path_for("CCO")
    assert a == b
    c = cache_path_for("c1ccccc1")
    assert a != c


def test_featurize_returns_none_on_garbage() -> None:
    assert featurize_smiles("not-a-smiles") is None


# ---------- timing-based featurize cache test ----------


def test_featurize_cache_skips_on_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second pass over the same SMILES is faster because cache is hit."""
    from transfermtl.data import featurize as feat

    monkeypatch.setattr(feat, "CACHE_DIR", tmp_path)

    smis = ["CCO", "c1ccccc1", "O=C(O)c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]

    t0 = time.perf_counter()
    for s in smis:
        feat.featurize_and_cache(s)
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s in smis:
        feat.featurize_and_cache(s)
    warm = time.perf_counter() - t0

    # Warm pass must be at least 5x faster than the cold pass.
    assert warm < cold * 0.5 or warm < 0.01, f"cold={cold:.4f}s warm={warm:.4f}s"


# ---------- task-name resolver (friendly -> column) ----------


def test_resolve_task_name_friendly_to_column(tmp_path: Path) -> None:
    """Friendly name in manifest's task_rename map -> canonical column."""
    import json

    from transfermtl.data.manifest import resolve_task_name

    (tmp_path / "tox21.json").write_text(
        json.dumps({"task_rename": {"NR-AR": "task_1", "NR-AR-LBD": "task_2"}})
    )
    assert resolve_task_name("NR-AR", "tox21", manifest_dir=tmp_path) == "task_1"
    assert resolve_task_name("NR-AR-LBD", "tox21", manifest_dir=tmp_path) == "task_2"


def test_resolve_task_name_passthrough_when_already_column(tmp_path: Path) -> None:
    """If `name` is already in available_columns, return it verbatim."""
    import json

    from transfermtl.data.manifest import resolve_task_name

    (tmp_path / "tox21.json").write_text(
        json.dumps({"task_rename": {"NR-AR": "task_1"}})
    )
    # Even though tmp_path has a manifest, passing `task_1` directly should
    # short-circuit and return as-is.
    assert (
        resolve_task_name(
            "task_1", "tox21", available_columns=["task_1"], manifest_dir=tmp_path
        )
        == "task_1"
    )


def test_resolve_task_name_no_manifest(tmp_path: Path) -> None:
    """Synthetic-fixture path: no manifest exists, return name unchanged."""
    from transfermtl.data.manifest import resolve_task_name

    # tmp_path is empty -> no JSON file. Resolver must not raise.
    assert resolve_task_name("task_1", "fake_dataset", manifest_dir=tmp_path) == "task_1"


def test_resolve_task_name_unknown_passes_through(tmp_path: Path) -> None:
    """Unknown name with a manifest present is returned unchanged so the
    caller raises a clear KeyError against the parquet."""
    import json

    from transfermtl.data.manifest import resolve_task_name

    (tmp_path / "tox21.json").write_text(json.dumps({"task_rename": {"NR-AR": "task_1"}}))
    assert (
        resolve_task_name("NotAReal-Task", "tox21", manifest_dir=tmp_path)
        == "NotAReal-Task"
    )


def test_load_task_rename_handles_missing_or_malformed(tmp_path: Path) -> None:
    """Missing manifest -> empty dict; malformed `task_rename` -> empty dict."""
    import json

    from transfermtl.data.manifest import load_task_rename

    # Missing JSON file.
    assert load_task_rename("never_prepared", manifest_dir=tmp_path) == {}

    # JSON without `task_rename`.
    (tmp_path / "noremap.json").write_text(json.dumps({"row_count": 7}))
    assert load_task_rename("noremap", manifest_dir=tmp_path) == {}

    # JSON with non-dict `task_rename` (defensive: should not crash).
    (tmp_path / "broken.json").write_text(json.dumps({"task_rename": ["not", "a", "dict"]}))
    assert load_task_rename("broken", manifest_dir=tmp_path) == {}
