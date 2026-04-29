"""Deterministic seeding for torch + numpy + cuda + cuDNN."""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np


def set_seed(seed: int) -> int:
    """Seed Python, NumPy, and (if available) torch + CUDA deterministically.

    Returns the active seed so callers can log it.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    with contextlib.suppress(ImportError):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with contextlib.suppress(AttributeError, RuntimeError):
            torch.use_deterministic_algorithms(True, warn_only=True)

    return seed
