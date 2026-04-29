"""Generate the synthetic 2-task fixture used across the test suite.

Layout (deterministic, seed=12345):
    - 200 molecules, 2 regions x 100 molecules each
    - 40 distinct scaffolds (20 per region, 5 molecules per scaffold)
    - Task 1: random binary, 50% positive
    - Task 2:
        Region 0: identical to task 1  -> aligned
        Region 1: opposite of task 1   -> opposed
    - Scaffold-stratified 70/15/15 split (14/3/3 scaffolds per region)

Outputs:
    tests/synthetic_fixture/data.parquet      -> SplitSchema-valid
    tests/synthetic_fixture/partition.parquet -> PartitionSchema-valid

Both files are intentionally checked in (gitignore excludes them; CI rebuilds
on first test run via the pytest fixture).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transfermtl.utils.io import write_parquet
from transfermtl.utils.schemas import PartitionSchema, SplitSchema

FIXTURE_DIR = Path(__file__).resolve().parent
DATA_PATH = FIXTURE_DIR / "data.parquet"
PARTITION_PATH = FIXTURE_DIR / "partition.parquet"

SEED = 12345
N_PER_REGION = 100
N_SCAFF_PER_REGION = 20
MOLS_PER_SCAFF = N_PER_REGION // N_SCAFF_PER_REGION  # 5

# Per-region split assignments at scaffold granularity: 14 train / 3 val / 3 test
# -> 70/15/15 mol counts per region, scaffold-disjoint across splits.
SPLIT_TEMPLATE = ["train"] * 14 + ["val"] * 3 + ["test"] * 3


def build_fixture(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two fixture parquets and return the in-memory frames."""
    if not force and DATA_PATH.exists() and PARTITION_PATH.exists():
        return (
            pd.read_parquet(DATA_PATH),
            pd.read_parquet(PARTITION_PATH),
        )

    rng = np.random.default_rng(SEED)

    rows: list[dict[str, object]] = []
    partition_rows: list[dict[str, object]] = []

    for region_idx, letter in enumerate(("A", "B")):
        # Per-region split shuffle (still deterministic via the rng).
        split_assign = list(SPLIT_TEMPLATE)
        rng.shuffle(split_assign)

        # Independent task_1 draws per region keep regions exchangeable.
        task_1 = rng.integers(0, 2, size=N_PER_REGION)
        # Region 0 (A) aligned; Region 1 (B) opposed.
        task_2 = task_1.copy() if region_idx == 0 else (1 - task_1)

        for s_idx in range(N_SCAFF_PER_REGION):
            scaff = f"scaff_{letter}_{s_idx:02d}"
            split_label = split_assign[s_idx]
            for m_idx in range(MOLS_PER_SCAFF):
                global_idx = s_idx * MOLS_PER_SCAFF + m_idx
                smiles = f"smi_{letter}_{s_idx:02d}_{m_idx}"
                rows.append(
                    {
                        "smiles": smiles,
                        "scaffold": scaff,
                        "split": split_label,
                        "task_1": float(task_1[global_idx]),
                        "task_2": float(task_2[global_idx]),
                    }
                )
                partition_rows.append(
                    {
                        "smiles": smiles,
                        "region_id": int(region_idx),
                    }
                )

    split_df = pd.DataFrame(rows)
    partition_df = pd.DataFrame(partition_rows)

    write_parquet(DATA_PATH, split_df, schema=SplitSchema)
    write_parquet(PARTITION_PATH, partition_df, schema=PartitionSchema)

    return split_df, partition_df


if __name__ == "__main__":
    s, p = build_fixture(force=True)
    print(f"split.parquet:    {DATA_PATH} ({len(s)} rows)")
    print(f"partition.parquet: {PARTITION_PATH} ({len(p)} rows)")
