"""Utilities for deterministic experiment setup."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
