"""Dataset interface for bitemporal image pairs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from scp.data.pair import ImagePair


class ChangeDataset(ABC):
    """Abstract dataset that yields validated bitemporal image pairs."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset identifier."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of pairs available."""

    @abstractmethod
    def __getitem__(self, index: int) -> ImagePair:
        """Return a pair by integer index."""

    @property
    def available_splits(self) -> tuple[str, ...]:
        """Return split names supported by loaders that expose split iteration."""
        return ()

    def iter_split(self, split: str) -> Iterator[ImagePair]:
        """Yield image pairs for a named split.

        Raises:
            NotImplementedError: If the dataset does not implement split streams.
        """
        raise NotImplementedError(f"{self.name} does not expose split streams")

    def iter_masks(self, split: str) -> Iterator[tuple[str, NDArray[np.uint8]]]:
        """Yield ground-truth masks for offline evaluation code.

        Raises:
            NotImplementedError: If the dataset does not expose ground-truth masks.
        """
        raise NotImplementedError(f"{self.name} does not expose mask streams")
