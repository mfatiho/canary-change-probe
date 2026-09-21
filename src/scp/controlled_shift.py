"""Deterministic sensor-inspired shifts for controlled audit experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from scp.data.dataset import ChangeDataset
from scp.data.pair import ImagePair

RgbImage = NDArray[np.uint8]
ImageTransform = Callable[[RgbImage], RgbImage]


@dataclass(frozen=True)
class ControlledShift:
    """One fully specified transformation applied to the after image."""

    name: str
    description: str
    transform: ImageTransform


class ControlledShiftDataset(ChangeDataset):
    """Apply one deterministic after-image shift while preserving masks."""

    def __init__(
        self,
        base_dataset: ChangeDataset,
        shift: ControlledShift,
    ) -> None:
        self._base_dataset = base_dataset
        self._shift = shift

    @property
    def name(self) -> str:
        """Return an identifier carrying the base dataset and shift."""
        return f"{self._base_dataset.name}:{self._shift.name}"

    @property
    def available_splits(self) -> tuple[str, ...]:
        """Expose the wrapped dataset's available splits."""
        return self._base_dataset.available_splits

    def __len__(self) -> int:
        """Return the wrapped dataset length."""
        return len(self._base_dataset)

    def __getitem__(self, index: int) -> ImagePair:
        """Return one transformed pair by index."""
        return self._transform_pair(self._base_dataset[index])

    def iter_split(self, split: str) -> Iterator[ImagePair]:
        """Yield transformed pairs from a named split."""
        for pair in self._base_dataset.iter_split(split):
            yield self._transform_pair(pair)

    def iter_masks(self, split: str) -> Iterator[tuple[str, NDArray[np.uint8]]]:
        """Yield unchanged masks aligned to transformed pair identifiers."""
        yield from self._base_dataset.iter_masks(split)

    def _transform_pair(self, pair: ImagePair) -> ImagePair:
        return ImagePair(
            before=pair.before.copy(),
            after=self._shift.transform(pair.after),
            pair_id=pair.pair_id,
        )


def controlled_shift_suite() -> tuple[ControlledShift, ...]:
    """Return the frozen controlled-shift suite used by the audit."""
    return (
        ControlledShift(
            name="identity",
            description="No input transformation.",
            transform=_identity,
        ),
        ControlledShift(
            name="registration_2px",
            description="After image translated two pixels right and down.",
            transform=lambda image: _translate_reflect(image, row_offset=2, col_offset=2),
        ),
        ControlledShift(
            name="radiometric_gain",
            description="After-image digital numbers use gain 1.10 and offset 5.",
            transform=lambda image: _gain_offset(image, gain=1.10, offset=5.0),
        ),
        ControlledShift(
            name="resolution_half",
            description="After image downsampled by two and bilinearly restored.",
            transform=lambda image: _resolution_reduction(image, factor=2),
        ),
    )


def _identity(image: RgbImage) -> RgbImage:
    return image.copy()


def _translate_reflect(
    image: RgbImage,
    row_offset: int,
    col_offset: int,
) -> RgbImage:
    if row_offset < 0 or col_offset < 0:
        raise ValueError("translation offsets must be non-negative")
    if row_offset >= image.shape[0] or col_offset >= image.shape[1]:
        raise ValueError("translation offsets must be smaller than the image")
    padded = np.pad(
        image,
        ((row_offset, 0), (col_offset, 0), (0, 0)),
        mode="reflect",
    )
    return np.asarray(padded[: image.shape[0], : image.shape[1]], dtype=np.uint8)


def _gain_offset(image: RgbImage, gain: float, offset: float) -> RgbImage:
    if gain <= 0:
        raise ValueError("radiometric gain must be positive")
    transformed = image.astype(np.float32) * gain + offset
    return np.clip(np.rint(transformed), 0, 255).astype(np.uint8)


def _resolution_reduction(image: RgbImage, factor: int) -> RgbImage:
    if factor <= 1:
        raise ValueError("resolution reduction factor must exceed one")
    height, width = image.shape[:2]
    reduced_size = (max(1, width // factor), max(1, height // factor))
    pil_image = Image.fromarray(image)
    reduced = pil_image.resize(reduced_size, resample=Image.Resampling.BILINEAR)
    restored = reduced.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(restored, dtype=np.uint8)


__all__ = [
    "ControlledShift",
    "ControlledShiftDataset",
    "controlled_shift_suite",
]
