"""Canary generator abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from scp.data import ImagePair


class CanaryGenerator(ABC):
    """Creates a deterministic synthetic change and returns its known mask."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Generator registry name."""

    @abstractmethod
    def generate(
        self, pair: ImagePair, rng: np.random.Generator
    ) -> tuple[ImagePair, NDArray[np.bool_]]:
        """Return a modified image pair and the inserted canary mask."""

