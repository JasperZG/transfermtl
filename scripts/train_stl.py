"""CLI entry point for STL training.

Usage:
    python scripts/train_stl.py --dataset tox21 --task NR-AR --seed 0

Reads the split parquet (A2) + featurization cache (A2), trains an STL model on
the requested task, and writes a checkpoint + test-set predictions parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from transfermtl.training import (
    STLArtifacts,
    TrainConfig,
    load_pyg_dataset,
    train_stl,
)
from transfermtl.utils import logging as wandb_logging
from transfermtl.utils.git import current_sha, dirty_tree_warning
from transfermtl.utils.io import read_parquet
from transfermtl.utils.schemas import SplitSchema

log = logging.getLogger("train_stl")

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "outputs" / "splits"
CACHE_DIR = REPO_ROOT / "outputs" / "cache" / "featurized"
CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints"
PRED_DIR = REPO_ROOT / "outputs" / "predictions"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("git=%s dirty=%s", current_sha(), dirty_tree_warning())

    cfg = TrainConfig(max_epochs=args.max_epochs, batch_size=args.batch_size)

    split_path = SPLITS_DIR / args.dataset / "split.parquet"
    split_df = read_parquet(split_path, schema=SplitSchema)

    train = load_pyg_dataset(split_df, "train", [args.task], cache_dir=CACHE_DIR)
    val = load_pyg_dataset(split_df, "val", [args.task], cache_dir=CACHE_DIR)
    test = load_pyg_dataset(split_df, "test", [args.task], cache_dir=CACHE_DIR)
    log.info("train=%d val=%d test=%d", len(train), len(val), len(test))

    wandb_logging.init(
        config={"dataset": args.dataset, "task": args.task, "seed": args.seed, **cfg.__dict__},
        run_name=f"stl/{args.dataset}/{args.task}/seed{args.seed}",
        tags=["stl", args.dataset, args.task],
    )

    ckpt_path = CKPT_DIR / args.dataset / "stl" / args.task / f"seed{args.seed}.pt"
    pred_path = PRED_DIR / args.dataset / "stl" / args.task / f"seed{args.seed}.parquet"

    art: STLArtifacts = train_stl(
        train_data=train,
        val_data=val,
        task=args.task,
        seed=args.seed,
        cfg=cfg,
        device=args.device,
        checkpoint_path=ckpt_path,
        test_data=test,
        predictions_path=pred_path,
    )

    log.info(
        "best_val_loss=%.4f best_epoch=%d ckpt=%s pred=%s",
        art.outcome.best_val_loss,
        art.outcome.best_epoch,
        art.checkpoint_path,
        art.predictions_path,
    )
    wandb_logging.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
