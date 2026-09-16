"""Deterministic seeding helpers (PRD 15 - Reproducibility)."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(False)  # keep CPU kernels fast
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(*parts: int, base: Optional[int] = None) -> int:
    """Stable child seed from a (seed, round, client) tuple."""
    value = 0 if base is None else int(base)
    for part in parts:
        value = (value * 1_000_003 + int(part)) % (2**31 - 1)
    return value


def rng(*parts: int, base: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts, base=base))
