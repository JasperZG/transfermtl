"""CLI entry point for the preprocessing pipeline.

Usage:
    python scripts/prepare_dataset.py --dataset tox21
    python scripts/prepare_dataset.py --dataset sider --force
    python scripts/prepare_dataset.py --dataset tox21 --no-featurize  # skip per-mol .pt cache
"""

from __future__ import annotations

import argparse
import logging
import sys

from transfermtl.data.datasets import prepare_dataset
from transfermtl.utils.git import current_sha, dirty_tree_warning

log = logging.getLogger("prepare_dataset")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["tox21", "sider"])
    parser.add_argument("--force", action="store_true", help="ignore idempotency cache")
    parser.add_argument(
        "--no-featurize",
        action="store_true",
        help="skip per-molecule PyG featurization cache",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log.info("git=%s dirty=%s", current_sha(), dirty_tree_warning())

    out = prepare_dataset(
        args.dataset,
        force=args.force,
        do_featurize=not args.no_featurize,
    )
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
