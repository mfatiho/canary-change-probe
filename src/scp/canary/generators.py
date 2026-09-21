"""Pure NumPy deterministic canary change generators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scp.canary.base import CanaryGenerator
from scp.canary.bank import CanaryBank, load_canary_bank
from scp.data import ImagePair
from scp.utils import ConfigError

# Defaults approximate 8 m building and 3 m road footprints at 0.5 m/px.
DEFAULT_CANARY_SIZE = 16
ROAD_WIDTH = 6


@dataclass(frozen=True)
class SmallBuildingAdd(CanaryGenerator):
    """Add a bright square canary to the after image."""

    size: int = DEFAULT_CANARY_SIZE
    intensity: float = 1.0

    @property
    def name(self) -> str:
        """Generator registry name."""
        return "small_building_add"

    def generate(
        self, pair: ImagePair, rng: np.random.Generator
    ) -> tuple[ImagePair, NDArray[np.bool_]]:
        """Insert a bright square into the after image."""
        row, col = _random_top_left(pair.spatial_shape, self.size, rng)
        mask = _square_mask(pair.spatial_shape, row, col, self.size)
        after = pair.after.copy()
        _assign_masked_pixels(after, mask, self.intensity)
        return pair.copy_with(after=after), mask


@dataclass(frozen=True)
class BuildingRemove(CanaryGenerator):
    """Darken a square region to simulate removal in the after image."""

    size: int = DEFAULT_CANARY_SIZE
    replacement: float = 0.0

    @property
    def name(self) -> str:
        """Generator registry name."""
        return "building_remove"

    def generate(
        self, pair: ImagePair, rng: np.random.Generator
    ) -> tuple[ImagePair, NDArray[np.bool_]]:
        """Remove an existing bright-looking square from the after image."""
        row, col = _random_top_left(pair.spatial_shape, self.size, rng)
        mask = _square_mask(pair.spatial_shape, row, col, self.size)
        after = pair.after.copy()
        _assign_masked_pixels(after, mask, self.replacement)
        return pair.copy_with(after=after), mask


CROP_RADIOMETRY_BLEND = 0.5
"""Fraction of mean/std alignment applied to a pasted crop."""


@dataclass(frozen=True)
class BuildingCropAdd(CanaryGenerator):
    """Paste a real source building crop into the after image."""

    bank_path: Path | str | None
    radiometry_blend: float = CROP_RADIOMETRY_BLEND

    def __post_init__(self) -> None:
        """Validate that a crop bank is configured."""
        if self.bank_path is None:
            raise ConfigError("canary.bank_path must be configured for building_crop_add")
        if not Path(self.bank_path).is_file():
            raise ConfigError(f"canary bank does not exist: {self.bank_path}")
        if not 0.0 <= self.radiometry_blend <= 1.0:
            raise ConfigError("building_crop_add.radiometry_blend must be in [0, 1]")

    @property
    def name(self) -> str:
        """Generator registry name."""
        return "building_crop_add"

    def generate(
        self, pair: ImagePair, rng: np.random.Generator
    ) -> tuple[ImagePair, NDArray[np.bool_]]:
        """Paste a radiometry-aligned real building component into the after image."""
        bank = _cached_canary_bank(Path(self.bank_path))
        crop, crop_mask = _select_crop_for_pair(bank.crops, bank.masks, pair.spatial_shape, rng)
        crop_height, crop_width = crop_mask.shape
        row, col = _random_top_left_for_shape(pair.spatial_shape, (crop_height, crop_width), rng)
        after = pair.after.copy()
        matched_crop = align_crop_radiometry(
            crop,
            after,
            blend=self.radiometry_blend,
        )
        reference = after[row : row + crop_height, col : col + crop_width]
        if _masked_arrays_equal(reference, matched_crop, crop_mask):
            matched_crop = _mean_shift_crop(crop, reference)
        full_mask = np.zeros(pair.spatial_shape, dtype=bool)
        full_mask[row : row + crop_height, col : col + crop_width] = crop_mask
        target_patch = after[row : row + crop_height, col : col + crop_width]
        _assign_masked_array(target_patch, crop_mask, matched_crop)
        return pair.copy_with(after=after), full_mask


@dataclass(frozen=True)
class BuildingRemoveFill(CanaryGenerator):
    """Replace a component-sized region with sampled surrounding texture."""

    size: int = DEFAULT_CANARY_SIZE
    ring_width: int = 6

    @property
    def name(self) -> str:
        """Generator registry name."""
        return "building_remove_fill"

    def generate(
        self, pair: ImagePair, rng: np.random.Generator
    ) -> tuple[ImagePair, NDArray[np.bool_]]:
        """Fill a removal mask with non-solid texture sampled from nearby pixels."""
        row, col = _salient_top_left(pair.after, self.size, rng)
        mask = _square_mask(pair.spatial_shape, row, col, self.size)
        after = pair.after.copy()
        ring = _ring_mask(pair.spatial_shape, row, col, self.size, self.ring_width)
        source_pixels = _texture_source_pixels(after, mask, ring, rng)
        replacement = _sample_texture(source_pixels, int(mask.sum()), rng, after.dtype)
        if after.ndim == 2:
            after[mask] = replacement
        else:
            after[mask, :] = replacement
        return pair.copy_with(after=after), mask


@dataclass(frozen=True)
class LinearRoadAdd(CanaryGenerator):
    """Add a simple horizontal or vertical linear road canary."""

    width: int = ROAD_WIDTH
    intensity: float = 0.8

    @property
    def name(self) -> str:
        """Generator registry name."""
        return "linear_road_add"

    def generate(
        self, pair: ImagePair, rng: np.random.Generator
    ) -> tuple[ImagePair, NDArray[np.bool_]]:
        """Draw a deterministic random-axis road into the after image."""
        height, width = pair.spatial_shape
        mask = np.zeros(pair.spatial_shape, dtype=bool)
        if bool(rng.integers(0, 2)):
            start = int(rng.integers(0, max(1, height - self.width + 1)))
            mask[start : start + self.width, :] = True
        else:
            start = int(rng.integers(0, max(1, width - self.width + 1)))
            mask[:, start : start + self.width] = True
        after = pair.after.copy()
        _assign_masked_pixels(after, mask, self.intensity)
        return pair.copy_with(after=after), mask


@dataclass(frozen=True)
class TextureReplace(CanaryGenerator):
    """Replace a square region with seeded random texture in the after image."""

    size: int = DEFAULT_CANARY_SIZE

    @property
    def name(self) -> str:
        """Generator registry name."""
        return "texture_replace"

    def generate(
        self, pair: ImagePair, rng: np.random.Generator
    ) -> tuple[ImagePair, NDArray[np.bool_]]:
        """Insert a random texture patch into the after image."""
        row, col = _random_top_left(pair.spatial_shape, self.size, rng)
        mask = _square_mask(pair.spatial_shape, row, col, self.size)
        after = pair.after.copy()
        texture_shape = (self.size, self.size, *after.shape[2:])
        texture = rng.random(texture_shape) * _intensity_scale(after)
        after[row : row + self.size, col : col + self.size] = texture.astype(after.dtype)
        return pair.copy_with(after=after), mask


@lru_cache(maxsize=8)
def _cached_canary_bank(bank_path: Path) -> CanaryBank:
    """Load and cache a canary bank keyed by path.

    `BuildingCropAdd.generate` runs once per canary per pair, so loading the
    bank fresh every call would re-read and re-deserialize the same `.npz`
    thousands of times per run. The bank file is written once up front and
    treated as read-only for the rest of the pipeline, so caching by path is
    safe.
    """
    return load_canary_bank(bank_path)


def _random_top_left(
    spatial_shape: tuple[int, int], size: int, rng: np.random.Generator
) -> tuple[int, int]:
    if size <= 0:
        raise ValueError("canary size must be positive")
    height, width = spatial_shape
    if size > height or size > width:
        raise ValueError("canary size must fit within the image")
    row = int(rng.integers(0, height - size + 1))
    col = int(rng.integers(0, width - size + 1))
    return row, col


def _random_top_left_for_shape(
    spatial_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[int, int]:
    crop_height, crop_width = crop_shape
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("crop shape must be positive")
    height, width = spatial_shape
    if crop_height > height or crop_width > width:
        raise ValueError("crop must fit within the image")
    row = int(rng.integers(0, height - crop_height + 1))
    col = int(rng.integers(0, width - crop_width + 1))
    return row, col


def _square_mask(spatial_shape: tuple[int, int], row: int, col: int, size: int) -> NDArray[np.bool_]:
    mask = np.zeros(spatial_shape, dtype=bool)
    mask[row : row + size, col : col + size] = True
    return mask


SALIENT_CANDIDATE_COUNT = 16
"""Candidate windows sampled when choosing a removal site; the highest-variance
candidate is removed so the canary deletes visible structure, not empty ground."""


def align_crop_radiometry(
    crop: NDArray[np.floating] | NDArray[np.integer],
    reference: NDArray[np.floating] | NDArray[np.integer],
    blend: float = CROP_RADIOMETRY_BLEND,
) -> NDArray[np.floating] | NDArray[np.integer]:
    """Shift crop channel statistics part-way toward a reference image.

    Unlike full histogram matching, mean/std alignment blended at `blend`
    preserves the crop's internal structure and most of its natural contrast,
    so the pasted object stays visible against its new background.
    """
    crop_array = np.asarray(crop, dtype=np.float32)
    reference_array = np.asarray(reference, dtype=np.float32)
    if (crop_array.ndim == 2) != (reference_array.ndim == 2):
        raise ValueError("crop and reference must both be grayscale or both be multi-channel")
    if crop_array.ndim == 2:
        aligned = _align_single_channel(crop_array, reference_array, blend)
    else:
        aligned = np.stack(
            [
                _align_single_channel(
                    crop_array[..., channel], reference_array[..., channel], blend
                )
                for channel in range(crop_array.shape[-1])
            ],
            axis=-1,
        )
    return _cast_to_dtype(aligned, np.asarray(crop).dtype)


def _align_single_channel(
    channel: NDArray[np.float32],
    reference: NDArray[np.float32],
    blend: float,
) -> NDArray[np.float32]:
    channel_std = float(channel.std()) or 1.0
    reference_std = float(reference.std()) or 1.0
    moved = (channel - float(channel.mean())) / channel_std * reference_std
    moved = moved + float(reference.mean())
    return (1.0 - blend) * channel + blend * moved


def _salient_top_left(
    image: NDArray[np.floating] | NDArray[np.integer],
    size: int,
    rng: np.random.Generator,
    candidate_count: int = SALIENT_CANDIDATE_COUNT,
) -> tuple[int, int]:
    """Pick the highest-variance window among seeded random candidates."""
    best_position: tuple[int, int] | None = None
    best_score = -1.0
    for _ in range(candidate_count):
        row, col = _random_top_left(image.shape[:2], size, rng)
        window = np.asarray(image[row : row + size, col : col + size], dtype=np.float32)
        score = float(window.std())
        if score > best_score:
            best_score = score
            best_position = (row, col)
    assert best_position is not None
    return best_position


def histogram_match_channels(
    image: NDArray[np.floating] | NDArray[np.integer],
    reference: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.floating] | NDArray[np.integer]:
    """Return `image` with each channel's empirical CDF matched to `reference`."""
    image_array = np.asarray(image)
    reference_array = np.asarray(reference)
    if image_array.shape != reference_array.shape:
        raise ValueError(
            f"image and reference shapes must match: {image_array.shape} vs {reference_array.shape}"
        )
    if image_array.ndim == 2:
        matched = _match_single_channel(image_array, reference_array)
    else:
        channel_matches = [
            _match_single_channel(image_array[..., channel], reference_array[..., channel])
            for channel in range(image_array.shape[-1])
        ]
        matched = np.stack(channel_matches, axis=-1)
    return _cast_to_dtype(matched, image_array.dtype)


def _match_single_channel(
    image: NDArray[np.floating] | NDArray[np.integer],
    reference: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.float64]:
    source_values, source_inverse, source_counts = np.unique(
        image.ravel(), return_inverse=True, return_counts=True
    )
    reference_values, reference_counts = np.unique(reference.ravel(), return_counts=True)
    source_quantiles = np.cumsum(source_counts).astype(np.float64)
    source_quantiles /= source_quantiles[-1]
    reference_quantiles = np.cumsum(reference_counts).astype(np.float64)
    reference_quantiles /= reference_quantiles[-1]
    matched_values = np.interp(source_quantiles, reference_quantiles, reference_values)
    return matched_values[source_inverse].reshape(image.shape)


def _select_crop_for_pair(
    crops: tuple[NDArray[np.floating] | NDArray[np.integer], ...],
    masks: tuple[NDArray[np.bool_], ...],
    spatial_shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating] | NDArray[np.integer], NDArray[np.bool_]]:
    height, width = spatial_shape
    candidates = [
        (crop, mask)
        for crop, mask in zip(crops, masks, strict=True)
        if mask.shape[0] <= height and mask.shape[1] <= width and np.any(mask)
    ]
    if not candidates:
        raise ConfigError("canary bank contains no crops that fit the target image")
    return candidates[int(rng.integers(0, len(candidates)))]


def _assign_masked_array(
    target: NDArray[np.floating] | NDArray[np.integer],
    mask: NDArray[np.bool_],
    values: NDArray[np.floating] | NDArray[np.integer],
) -> None:
    if target.ndim == 2:
        target[mask] = values[mask]
        return
    target[mask, :] = values[mask, :]


def _masked_arrays_equal(
    target: NDArray[np.floating] | NDArray[np.integer],
    values: NDArray[np.floating] | NDArray[np.integer],
    mask: NDArray[np.bool_],
) -> bool:
    if target.ndim == 2:
        return bool(np.array_equal(target[mask], values[mask]))
    return bool(np.array_equal(target[mask, :], values[mask, :]))


def _mean_shift_crop(
    crop: NDArray[np.floating] | NDArray[np.integer],
    reference: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.floating] | NDArray[np.integer]:
    crop_float = np.asarray(crop, dtype=np.float64)
    reference_float = np.asarray(reference, dtype=np.float64)
    axes = (0, 1)
    shifted = crop_float - np.mean(crop_float, axis=axes, keepdims=True)
    shifted += np.mean(reference_float, axis=axes, keepdims=True)
    if np.allclose(shifted, reference_float):
        shifted = crop_float
    return _cast_to_dtype(shifted, np.asarray(crop).dtype)


def _ring_mask(
    spatial_shape: tuple[int, int],
    row: int,
    col: int,
    size: int,
    ring_width: int,
) -> NDArray[np.bool_]:
    if ring_width <= 0:
        raise ValueError("ring_width must be positive")
    height, width = spatial_shape
    outer_top = max(0, row - ring_width)
    outer_left = max(0, col - ring_width)
    outer_bottom = min(height, row + size + ring_width)
    outer_right = min(width, col + size + ring_width)
    ring = np.zeros(spatial_shape, dtype=bool)
    ring[outer_top:outer_bottom, outer_left:outer_right] = True
    ring[row : row + size, col : col + size] = False
    return ring


def _texture_source_pixels(
    image: NDArray[np.floating] | NDArray[np.integer],
    mask: NDArray[np.bool_],
    ring: NDArray[np.bool_],
    rng: np.random.Generator,
) -> NDArray[np.floating] | NDArray[np.integer]:
    if np.any(ring):
        return image[ring]
    outside = ~mask
    if np.any(outside):
        return image[outside]
    fallback_shape = (max(2, int(mask.sum())), *image.shape[2:])
    return (rng.random(fallback_shape) * _intensity_scale(image)).astype(image.dtype)


def _sample_texture(
    source_pixels: NDArray[np.floating] | NDArray[np.integer],
    count: int,
    rng: np.random.Generator,
    dtype: np.dtype,
) -> NDArray[np.floating] | NDArray[np.integer]:
    source_count = int(source_pixels.shape[0])
    if source_count == 0:
        raise ValueError("source_pixels must not be empty")
    indices = rng.integers(0, source_count, size=count)
    return _cast_to_dtype(np.asarray(source_pixels)[indices], dtype)


def _assign_masked_pixels(
    image: NDArray[np.floating] | NDArray[np.integer], mask: NDArray[np.bool_], value: float
) -> None:
    """Assign a [0, 1] semantic intensity, scaled to the image's dtype range.

    Dataset images are uint8 (0-255); writing raw [0, 1] intensities into them
    produces near-black patches that detectors rightly ignore as shadow.
    """
    scaled = value * _intensity_scale(image)
    if image.ndim == 2:
        image[mask] = scaled
        return
    image[mask, :] = scaled


def _intensity_scale(image: NDArray[np.floating] | NDArray[np.integer]) -> float:
    """Return the dtype value range: iinfo max for integer images, 1.0 for float."""
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)
    return 1.0


def _cast_to_dtype(
    values: NDArray[np.floating] | NDArray[np.integer],
    dtype: np.dtype,
) -> NDArray[np.floating] | NDArray[np.integer]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)
