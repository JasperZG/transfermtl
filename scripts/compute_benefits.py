"""Compute regional transfer benefits Δ_ij(r) with hierarchical-bootstrap CIs.

Reads:
  outputs/predictions/{dataset}/stl/{task_i,task_j}/seed*.parquet  (A3)
  outputs/predictions/{dataset}/mtl/{task_i}_{task_j}/seed*.parquet (A3)
  outputs/partitions/{dataset}/{partition}.parquet                  (A5)
  outputs/splits/{dataset}/split.parquet                            (A2; for scaffold ids)

Writes:
  outputs/benefits/{dataset}/{task_i}_{task_j}/region_benefits.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from transfermtl.benefits.aggregate import aggregate_region_benefits, write_region_benefits
from transfermtl.utils.schemas import PartitionSchema, PredictionSchema, SplitSchema

log = logging.getLogger("compute_benefits")


def _load_predictions_per_seed(root: Path) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for p in sorted(root.glob("seed*.parquet")):
        seed = int(p.stem.replace("seed", ""))
        df = PredictionSchema.validate(pd.read_parquet(p))
        out[seed] = df
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-i", required=True)
    parser.add_argument("--task-j", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--task-type", choices=["clf", "reg"], default="clf")
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    split_df = SplitSchema.validate(
        pd.read_parquet(Path("outputs/splits") / args.dataset / "split.parquet")
    )
    partition = PartitionSchema.validate(
        pd.read_parquet(Path("outputs/partitions") / args.dataset / f"{args.partition}.parquet")
    )
    stl_i = _load_predictions_per_seed(
        Path("outputs/predictions") / args.dataset / "stl" / args.task_i
    )
    stl_j = _load_predictions_per_seed(
        Path("outputs/predictions") / args.dataset / "stl" / args.task_j
    )
    mtl = _load_predictions_per_seed(
        Path("outputs/predictions") / args.dataset / "mtl" / f"{args.task_i}_{args.task_j}"
    )

    benefits = aggregate_region_benefits(
        stl_i,
        stl_j,
        mtl,
        partition,
        split_df,
        task_i=args.task_i,
        task_j=args.task_j,
        task_type=args.task_type,
        n_iter=args.n_boot,
    )
    out = write_region_benefits(args.dataset, args.task_i, args.task_j, benefits)
    log.info("wrote %s (%d regions)", out, len(benefits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
