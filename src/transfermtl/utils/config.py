"""Hydra/OmegaConf wrappers + lock-file enforcement for configs/_shared/.

The pre-committed hyperparameter table (plan §2.21) lives in YAMLs under
configs/_shared/. _lock.yaml records sha256 of each file; loaders verify the
hashes match before returning a config. CI runs `--check-lock` to catch any
drift before a job spawns.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_DIR = REPO_ROOT / "configs" / "_shared"
LOCK_FILE = SHARED_DIR / "_lock.yaml"

# Files whose hashes are tracked. _lock.yaml itself is excluded.
LOCKED_FILES = (
    "encoder_gcn.yaml",
    "train_default.yaml",
    "bootstrap.yaml",
    "preprocess.yaml",
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def compute_lock() -> dict[str, str]:
    """Compute current sha256 for each tracked file."""
    return {name: sha256_of(SHARED_DIR / name) for name in LOCKED_FILES}


def load_lock() -> dict[str, str]:
    """Load the recorded sha256 map from _lock.yaml."""
    if not LOCK_FILE.exists():
        raise FileNotFoundError(f"Lock file missing: {LOCK_FILE}")
    with LOCK_FILE.open() as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("sha256", {}))


def verify_lock() -> tuple[bool, list[str]]:
    """Return (ok, mismatched_files)."""
    recorded = load_lock()
    actual = compute_lock()
    mismatches = [name for name in LOCKED_FILES if recorded.get(name) != actual.get(name)]
    return (not mismatches), mismatches


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config and enforce the lock file before returning.

    Mismatches between configs/_shared/_lock.yaml and the actual file checksums
    raise RuntimeError — this prevents post-hoc hyperparameter drift.
    """
    ok, mismatches = verify_lock()
    if not ok:
        raise RuntimeError(
            f"configs/_shared/_lock.yaml mismatch: {mismatches}. "
            "Frozen hyperparameters changed without updating the lock file."
        )
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {path}, got {type(cfg).__name__}")
    return cfg


def write_lock() -> Path:
    """Write _lock.yaml from the current shared file checksums.

    Used during Wave 1 setup; should not be invoked in normal operation.
    """
    data = {"sha256": compute_lock()}
    LOCK_FILE.write_text(yaml.safe_dump(data, sort_keys=True))
    return LOCK_FILE


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Config lock utilities.")
    parser.add_argument("--check-lock", action="store_true")
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args(argv)

    if args.write_lock:
        path = write_lock()
        print(f"Wrote {path}")
        return 0

    if args.check_lock:
        ok, mismatches = verify_lock()
        if ok:
            print("lock OK")
            return 0
        print(f"lock MISMATCH: {mismatches}", file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
