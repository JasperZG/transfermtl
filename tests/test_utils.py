"""Smoke tests for utils package: seeding, registry, schemas, io, lock file."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from transfermtl.utils import config as cfgmod
from transfermtl.utils.io import read_npy, read_parquet, write_npy, write_parquet
from transfermtl.utils.registry import REGISTRY, build, register
from transfermtl.utils.schemas import (
    GradientAffinitySchema,
    PartitionSchema,
    PredictionSchema,
    SplitSchema,
    pa,
)
from transfermtl.utils.seeding import set_seed

SchemaErrors = (pa.errors.SchemaError, pa.errors.SchemaErrors)


def test_seeding_deterministic() -> None:
    """Same seed -> identical numpy and torch draws."""
    set_seed(7)
    a = np.random.rand(8)
    set_seed(7)
    b = np.random.rand(8)
    np.testing.assert_array_equal(a, b)

    import torch

    set_seed(11)
    ta = torch.randn(8)
    set_seed(11)
    tb = torch.randn(8)
    assert torch.equal(ta, tb)


def test_registry_lookup() -> None:
    REGISTRY.pop("encoder", None)

    @register("encoder", "x")
    def factory(value: int = 3) -> dict[str, int]:
        return {"value": value}

    out = build("encoder", "x", value=5)
    assert out == {"value": 5}

    with pytest.raises(KeyError):
        build("encoder", "missing")


def test_schemas_validate_good() -> None:
    split = pd.DataFrame(
        {
            "smiles": ["A", "B"],
            "scaffold": ["s1", "s2"],
            "split": ["train", "test"],
            "task_1": [1.0, 0.0],
            "task_2": [0.0, np.nan],
        }
    )
    SplitSchema.validate(split)

    part = pd.DataFrame({"smiles": ["A", "B"], "region_id": [0, 1]})
    PartitionSchema.validate(part)

    pred = pd.DataFrame(
        {
            "smiles": ["A"],
            "task": ["t1"],
            "y_true": [1.0],
            "y_pred": [0.7],
            "seed": [0],
        }
    )
    PredictionSchema.validate(pred)


def test_schemas_reject_bad() -> None:
    # Missing column 'scaffold'
    bad = pd.DataFrame({"smiles": ["A"], "split": ["train"], "task_1": [1.0]})
    with pytest.raises(SchemaErrors):
        SplitSchema.validate(bad)

    # Wrong split label
    bad2 = pd.DataFrame(
        {
            "smiles": ["A"],
            "scaffold": ["s1"],
            "split": ["holdout"],
            "task_1": [1.0],
        }
    )
    with pytest.raises(SchemaErrors):
        SplitSchema.validate(bad2)

    # Extra column rejected by strict=True
    bad3 = pd.DataFrame(
        {
            "smiles": ["A"],
            "scaffold": ["s1"],
            "split": ["train"],
            "task_1": [1.0],
            "extra": [42],
        }
    )
    with pytest.raises(SchemaErrors):
        SplitSchema.validate(bad3)

    # Bad gradient checkpoint label
    bad4 = pd.DataFrame(
        {
            "region_id": [0],
            "G_ij": [0.5],
            "g_i_norm": [1.0],
            "g_j_norm": [1.0],
            "n_i_in_region": [50],
            "n_j_in_region": [50],
            "checkpoint_label": ["epoch10"],
            "seed": [0],
        }
    )
    with pytest.raises(SchemaErrors):
        GradientAffinitySchema.validate(bad4)


def test_lock_file_matches_shared() -> None:
    """The recorded sha256 in _lock.yaml equals sha256 of each file in _shared/."""
    recorded = yaml.safe_load(cfgmod.LOCK_FILE.read_text())["sha256"]
    for name in cfgmod.LOCKED_FILES:
        actual = hashlib.sha256((cfgmod.SHARED_DIR / name).read_bytes()).hexdigest()
        assert recorded[name] == actual, name

    ok, mismatches = cfgmod.verify_lock()
    assert ok, mismatches


def test_io_roundtrip_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "smiles": ["A", "B"],
            "scaffold": ["s1", "s2"],
            "split": ["train", "val"],
            "task_1": [0.0, 1.0],
        }
    )
    out = write_parquet(tmp_path / "x.parquet", df, schema=SplitSchema)
    back = read_parquet(out, schema=SplitSchema)
    pd.testing.assert_frame_equal(
        back.sort_index(axis=1).reset_index(drop=True),
        df.sort_index(axis=1).reset_index(drop=True),
        check_dtype=False,
    )


def test_io_roundtrip_npy(tmp_path: Path) -> None:
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = write_npy(tmp_path / "x.npy", arr)
    back = read_npy(out)
    np.testing.assert_array_equal(arr, back)


def test_load_config_enforces_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config() raises if the lock file disagrees with the shared YAMLs."""
    cfg = cfgmod.load_config(cfgmod.SHARED_DIR / "encoder_gcn.yaml")
    assert cfg["name"] == "gcn"

    fake_lock = tmp_path / "fake_lock.yaml"
    fake_lock.write_text(yaml.safe_dump({"sha256": dict.fromkeys(cfgmod.LOCKED_FILES, "0" * 64)}))
    monkeypatch.setattr(cfgmod, "LOCK_FILE", fake_lock)
    with pytest.raises(RuntimeError):
        cfgmod.load_config(cfgmod.SHARED_DIR / "encoder_gcn.yaml")


def test_git_sha_is_string() -> None:
    from transfermtl.utils.git import current_sha

    sha = current_sha()
    assert isinstance(sha, str)
    assert sha != ""
