"""Tests for deterministic sensor-inspired controlled shifts."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from scp.data import (
    ChangeDataset,
    ControlledShiftDataset,
    ImagePair,
    controlled_shift_suite,
)


class _OnePairDataset(ChangeDataset):
    @property
    def name(self) -> str:
        return "fixture"

    @property
    def available_splits(self) -> tuple[str, ...]:
        return ("test",)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> ImagePair:
        if index != 0:
            raise IndexError(index)
        return next(self.iter_split("test"))

    def iter_split(self, split: str) -> Iterator[ImagePair]:
        if split != "test":
            raise ValueError(split)
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        yield ImagePair(before=image, after=image, pair_id="pair")

    def iter_masks(self, split: str) -> Iterator[tuple[str, NDArray[np.uint8]]]:
        if split != "test":
            raise ValueError(split)
        yield "pair", np.ones((8, 8), dtype=np.uint8)


def test_controlled_shift_suite_preserves_shape_ids_and_masks() -> None:
    """Every frozen shift should preserve the paired evaluation contract."""
    base = _OnePairDataset()

    for shift in controlled_shift_suite():
        dataset = ControlledShiftDataset(base, shift)
        pair = next(dataset.iter_split("test"))
        mask_id, mask = next(dataset.iter_masks("test"))
        assert pair.pair_id == mask_id == "pair"
        assert pair.before.shape == pair.after.shape == (8, 8, 3)
        assert mask.shape == (8, 8)


def test_nonidentity_shifts_modify_after_image_only() -> None:
    """Controlled shifts must leave the before image unchanged."""
    base_pair = next(_OnePairDataset().iter_split("test"))

    for shift in controlled_shift_suite()[1:]:
        shifted_pair = next(
            ControlledShiftDataset(_OnePairDataset(), shift).iter_split("test")
        )
        assert np.array_equal(shifted_pair.before, base_pair.before)
        assert not np.array_equal(shifted_pair.after, base_pair.after)
