"""Git SHA + dirty-tree capture for run metadata."""

from __future__ import annotations

import subprocess


def current_sha() -> str:
    """Return the current HEAD sha, or 'unknown' if git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def dirty_tree_warning() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
