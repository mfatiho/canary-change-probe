"""Bitemporal image-pair value object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ImagePair:
    """Validated pair of same-shaped before and after images."""

    before: NDArray[np.floating] | NDArray[np.integer]
    after: NDArray[np.floating] | NDArray[np.integer]
    pair_id: str | None = None

    def __post_init__(self) -> None:
        """Validate array shape compatibility."""
        if not isinstance(self.before, np.ndarray) or not isinstance(self.after, np.ndarray):
            raise TypeError("before and after must be NumPy arrays")
        if self.before.shape != self.after.shape:
            raise ValueError("before and after images must have the same shape")
        if self.before.ndim < 2:
            raise ValueError("image arrays must have at least two spatial dimensions")
        if 0 in self.before.shape[:2]:
            raise ValueError("image arrays must have non-empty spatial dimensions")

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return the height and width shared by both images."""
        return int(self.before.shape[0]), int(self.before.shape[1])

    def copy_with(
        self,
        before: NDArray[np.floating] | NDArray[np.integer] | None = None,
        after: NDArray[np.floating] | NDArray[np.integer] | None = None,
        pair_id: str | None = None,
    ) -> "ImagePair":
        """Create a validated copy with selected fields replaced."""
        return ImagePair(
            before=self.before if before is None else before,
            after=self.after if after is None else after,
            pair_id=self.pair_id if pair_id is None else pair_id,
        )
