"""Quick diagnostics for the pilot's scaffold partitioning + benefit coverage.

For each dataset, prints:
1. Region size distribution (total compounds per region)
2. Region x split crosstab (does the scaffold-stratified split distribute
   test compounds evenly across regions?)
3. For one example pair: how many of each region's compounds actually have
   STL predictions (i.e., made it past featurization + NaN-label filtering)

Run after stages 1-5 of launch_pilot.sh have completed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _example_pair(bench_dir: Path) -> tuple[str, str, str]:
    """Pick the first benefit dir and split its name into (pair, task_i, task_j).

    Pair dir name is `{task_i}_{task_j}`. Both task names can contain
    underscores (tox21: `NR-AR-LBD`) and spaces (sider). The split is by the
    rightmost underscore that separates the two tasks; for tox21 the first
    underscore works since task names don't contain `_`. We use rsplit to
    be conservative.
    """
    pair = sorted(p.name for p in bench_dir.iterdir() if p.is_dir())[0]
    task_i, task_j = pair.split("_", 1)
    return pair, task_i, task_j


def main() -> None:
    for ds in ("tox21", "sider"):
        print(f"\n========== {ds} ==========")
        part = pd.read_parquet(f"outputs/partitions/{ds}/scaffold_M5.parquet")
        split = pd.read_parquet(f"outputs/splits/{ds}/split.parquet")

        sizes = part.groupby("region_id").size()
        print(f"\nRegion sizes (total={sizes.sum()}):")
        print(sizes.to_string())
        print(f"share: {(sizes / sizes.sum()).round(3).to_dict()}")

        merged = part.merge(split[["smiles", "split"]], on="smiles")
        print("\nRegion x split:")
        print(merged.groupby(["region_id", "split"]).size().unstack(fill_value=0))

        bench_dir = Path(f"outputs/benefits/{ds}")
        if not bench_dir.exists():
            print(f"\n(no benefits dir at {bench_dir}; skipping prediction coverage)")
            continue
        pair, task_i, _task_j = _example_pair(bench_dir)
        print(f"\nExample pair {pair}: STL ({task_i}) prediction coverage per region")
        pred = pd.read_parquet(f"outputs/predictions/{ds}/stl/{task_i}/seed0.parquet")
        joined = part.merge(pred[["smiles", "y_pred"]], on="smiles", how="left")
        cov = joined.groupby("region_id")["y_pred"].agg(["count", "size"])
        cov.columns = ["with_pred", "total"]
        cov["coverage"] = (cov["with_pred"] / cov["total"]).round(3)
        print(cov.to_string())


if __name__ == "__main__":
    main()
