"""Shared pytest fixtures.

Wave 2+ agents request these fixtures by name in their unit tests so every
module is exercised on the same 200-mol synthetic dataset with known ground
truth.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.synthetic_fixture.build import build_fixture


@pytest.fixture(scope="session")
def synthetic_dataset() -> pd.DataFrame:
    """Combined SplitSchema frame: smiles, scaffold, split, task_1, task_2."""
    split_df, _ = build_fixture()
    return split_df


@pytest.fixture(scope="session")
def synthetic_split(synthetic_dataset: pd.DataFrame) -> pd.DataFrame:
    """Alias for `synthetic_dataset`; the fixture parquet is the split contract."""
    return synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_partition() -> pd.DataFrame:
    """PartitionSchema frame: smiles, region_id (0 = aligned, 1 = opposed)."""
    _, partition_df = build_fixture()
    return partition_df


@pytest.fixture(scope="session")
def synthetic_combined(
    synthetic_dataset: pd.DataFrame,
    synthetic_partition: pd.DataFrame,
) -> pd.DataFrame:
    """Convenience: split + partition merged on smiles for tests that need both."""
    return synthetic_dataset.merge(synthetic_partition, on="smiles", how="inner")
