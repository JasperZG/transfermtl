"""Compute regional gradient affinity G_ij(r) for one (dataset, pair, seed).

Reads:
  outputs/checkpoints/{dataset}/stl/{task}/seed{s}.pt        (A3)
  outputs/partitions/{dataset}/{partition}.parquet           (A5)
  outputs/cache/featurized/{16-char-sha}.pt                  (A2)

Writes:
  outputs/gradients/{dataset}/{task_i}_{task_j}/seed{s}/region_affinity.parquet

The script processes the `final` checkpoint by default. Pass --trajectory to
also compute the 0.8 and 0.6 snapshots (requires checkpoints saved at those
fractions, by name `..._final.pt`, `..._0.8.pt`, `..._0.6.pt`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from transfermtl.data.manifest import resolve_task_name
from transfermtl.gradients.affinity import cosine_affinity
from transfermtl.gradients.extract import compute_regional_gradient
from transfermtl.gradients.io import write_region_affinity
from transfermtl.training.data import load_pyg_dataset
from transfermtl.utils.schemas import PartitionSchema, SplitSchema

log = logging.getLogger("compute_gradients")


def _checkpoint_path(dataset: str, task: str, seed: int, label: str = "final") -> Path:
    base = Path("outputs/checkpoints") / dataset / "stl" / task
    if label == "final":
        return base / f"seed{seed}.pt"
    return base / f"seed{seed}_{label}.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-i", required=True)
    parser.add_argument("--task-j", required=True)
    parser.add_argument("--partition", required=True, help="e.g. scaffold_M5")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache-dir", default="outputs/cache/featurized")
    parser.add_argument("--max-subsample", type=int, default=500)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    split_path = Path("outputs/splits") / args.dataset / "split.parquet"
    partition_path = Path("outputs/partitions") / args.dataset / f"{args.partition}.parquet"

    split_df = SplitSchema.validate(pd.read_parquet(split_path))
    partition = PartitionSchema.validate(pd.read_parquet(partition_path))
    cache_dir = Path(args.cache_dir)

    # split.parquet stores tasks under canonical task_N columns; resolve the
    # friendly CLI names against the rename map so row[col] lookups succeed.
    col_i = resolve_task_name(args.task_i, args.dataset, list(split_df.columns))
    col_j = resolve_task_name(args.task_j, args.dataset, list(split_df.columns))

    rows: list[dict[str, object]] = []
    for region_id in sorted(partition["region_id"].unique()):
        region_smis = set(partition.loc[partition["region_id"] == region_id, "smiles"])
        region_split = split_df[split_df["smiles"].isin(region_smis)]
        if region_split.empty:
            log.warning("region %d empty in split", region_id)
            continue

        # Use train+val rows for gradient estimation (test compounds are held out).
        region_split = region_split[region_split["split"].isin(["train", "val"])]
        data_i = load_pyg_dataset(region_split, "train", [col_i], cache_dir=cache_dir)
        data_j = load_pyg_dataset(region_split, "train", [col_j], cache_dir=cache_dir)

        ckpt_i = _checkpoint_path(args.dataset, args.task_i, args.seed)
        ckpt_j = _checkpoint_path(args.dataset, args.task_j, args.seed)
        g_i, n_i, norm_i = compute_regional_gradient(
            ckpt_i, args.task_i, data_i, max_subsample=args.max_subsample, rng_seed=region_id
        )
        g_j, n_j, norm_j = compute_regional_gradient(
            ckpt_j, args.task_j, data_j, max_subsample=args.max_subsample, rng_seed=region_id
        )
        G = cosine_affinity(g_i, g_j)
        rows.append(
            {
                "region_id": int(region_id),
                "G_ij": float(G),
                "g_i_norm": float(norm_i),
                "g_j_norm": float(norm_j),
                "n_i_in_region": int(n_i),
                "n_j_in_region": int(n_j),
                "checkpoint_label": "final",
                "seed": int(args.seed),
            }
        )

    out = write_region_affinity(args.dataset, args.task_i, args.task_j, args.seed, rows)
    log.info("wrote %s (%d regions)", out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
