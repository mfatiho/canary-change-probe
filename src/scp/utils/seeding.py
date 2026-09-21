"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy RNGs and return an explicit NumPy generator.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A deterministic NumPy random generator.

    Raises:
        ValueError: If the seed is not a non-negative integer.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

