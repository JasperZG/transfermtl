"""Tests for the synthetic 2-task fixture."""

from __future__ import annotations

import pandas as pd

from tests.synthetic_fixture.build import DATA_PATH
from transfermtl.utils.schemas import SplitSchema


def test_fixture_shape(synthetic_dataset: pd.DataFrame) -> None:
    assert len(synthetic_dataset) == 200
    task_cols = [c for c in synthetic_dataset.columns if c.startswith("task_")]
    assert sorted(task_cols) == ["task_1", "task_2"]


def test_fixture_two_regions(synthetic_partition: pd.DataFrame) -> None:
    assert set(synthetic_partition["region_id"].unique()) == {0, 1}
    counts = synthetic_partition["region_id"].value_counts()
    assert counts[0] == 100
    assert counts[1] == 100


def test_fixture_ground_truth(synthetic_combined: pd.DataFrame) -> None:
    """Region 0: tasks identical (aligned). Region 1: tasks opposite (opposed)."""
    region_a = synthetic_combined[synthetic_combined["region_id"] == 0]
    region_b = synthetic_combined[synthetic_combined["region_id"] == 1]
    assert (region_a["task_1"] == region_a["task_2"]).all()
    assert (region_b["task_1"] != region_b["task_2"]).all()


def test_fixture_split_no_leakage(synthetic_dataset: pd.DataFrame) -> None:
    """Each scaffold appears in exactly one split."""
    grouped = synthetic_dataset.groupby("scaffold")["split"].nunique()
    assert (grouped == 1).all(), "scaffold leaked across splits"

    fracs = synthetic_dataset["split"].value_counts(normalize=True)
    assert fracs.get("train", 0) > 0.6
    assert fracs.get("test", 0) > 0.1


def test_fixture_passes_split_schema() -> None:
    """The on-disk fixture parquet validates against SplitSchema."""
    df = pd.read_parquet(DATA_PATH)
    SplitSchema.validate(df)
