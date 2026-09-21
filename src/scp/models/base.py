"""Model adapter abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from scp.data import ImagePair


class ModelLoadError(RuntimeError):
    """Raised when a third-party model backend cannot be loaded or used."""


class CDModelAdapter(ABC):
    """Abstract adapter that exposes probability-map prediction for a CD model."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter registry name."""

    @abstractmethod
    def predict(self, pair: ImagePair) -> NDArray[np.floating]:
        """Predict a change-probability map for an image pair."""

    def predict_batch(self, pairs: Sequence[ImagePair]) -> list[NDArray[np.float32]]:
        """Predict change-probability maps for image pairs in input order."""
        return [np.asarray(self.predict(pair), dtype=np.float32) for pair in pairs]
