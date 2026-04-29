"""Compute and persist region partitions for a dataset (plan §2.6, A5).

Usage:
    python scripts/compute_partitions.py --dataset tox21 --scheme scaffold --M 5
    python scripts/compute_partitions.py --dataset tox21 --scheme random \\
        --n-partitions 200

The latent and kNN schemes require an A3 all-task checkpoint (default path:
``outputs/checkpoints/{dataset}/all_task/seed0.pt``). Until A3 lands, those
schemes raise NotImplementedError when called from the CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys

from transfermtl.partition.io import write_partition
from transfermtl.partition.knn import compute_knn_partition
from transfermtl.partition.latent import compute_latent_partition
from transfermtl.partition.random_null import generate_random_partitions
from transfermtl.partition.scaffold import DEFAULT_N_MIN, compute_scaffold_partition
from transfermtl.utils.git import current_sha, dirty_tree_warning

log = logging.getLogger("compute_partitions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--scheme",
        required=True,
        choices=["scaffold", "latent", "knn", "random"],
    )
    parser.add_argument("--M", type=int, default=5, choices=[3, 5, 8, 10])
    parser.add_argument("--n-min", type=int, default=DEFAULT_N_MIN)
    parser.add_argument("--n-partitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("git=%s dirty=%s", current_sha(), dirty_tree_warning())

    if args.scheme == "scaffold":
        df = compute_scaffold_partition(args.dataset, M=args.M, n_min=args.n_min)
        path = write_partition(args.dataset, "scaffold", df, M=args.M)
        log.info("wrote %s (%d compounds, %d regions)", path, len(df), df["region_id"].nunique())

    elif args.scheme == "latent":
        df = compute_latent_partition(args.dataset, M=args.M, seed=args.seed)
        path = write_partition(args.dataset, "latent", df, M=args.M)
        log.info("wrote %s (%d compounds, %d regions)", path, len(df), df["region_id"].nunique())

    elif args.scheme == "knn":
        df = compute_knn_partition(args.dataset, M=args.M)
        path = write_partition(args.dataset, "knn", df, M=args.M)
        log.info("wrote %s (%d compounds, %d regions)", path, len(df), df["region_id"].nunique())

    elif args.scheme == "random":
        partitions = generate_random_partitions(
            args.dataset,
            n_partitions=args.n_partitions,
            seed=args.seed,
        )
        for b, df in enumerate(partitions):
            write_partition(args.dataset, "random", df, b=b)
        log.info("wrote %d random partitions for %s", len(partitions), args.dataset)

    return 0


if __name__ == "__main__":
    sys.exit(main())
