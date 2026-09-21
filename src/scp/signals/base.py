"""Passive signal abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from scp.data import ImagePair


class PassiveSignal(ABC):
    """Computes one scalar signal from an image pair and model output."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Signal registry name."""

    @abstractmethod
    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute a scalar signal value."""

