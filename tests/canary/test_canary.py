import numpy as np
from pathlib import Path

import pytest

from scp.canary import (
    BuildingCropAdd,
    BuildingRemoveFill,
    SmallBuildingAdd,
    get_canary_generator,
)
from scp.canary.generators import align_crop_radiometry, histogram_match_channels
from scp.data import ImagePair
from scp.utils import ConfigError

MIN_VISIBLE_CANARY_DELTA = 0.2
"""Minimum mean absolute in-mask change (float image scale) for a canary to count
as visible; camouflaged canaries defeat the probe's purpose."""


def make_pair() -> ImagePair:
    image = np.full((16, 16, 3), 0.25, dtype=np.float32)
    image[2:8, 2:8] = 0.9
    return ImagePair(before=image.copy(), after=image.copy())


def assert_canary_output(generator_name: str) -> None:
    rng = np.random.default_rng(7)
    modified_pair, mask = get_canary_generator(generator_name).generate(make_pair(), rng)

    assert modified_pair.before.shape == make_pair().before.shape
    assert modified_pair.after.shape == make_pair().after.shape
    assert mask.shape == make_pair().spatial_shape
    assert mask.dtype == np.bool_
    assert bool(mask.any())


def test_small_building_add_generates_masked_change() -> None:
    assert_canary_output("small_building_add")


def test_building_remove_generates_masked_change() -> None:
    assert_canary_output("building_remove")


def test_linear_road_add_generates_masked_change() -> None:
    assert_canary_output("linear_road_add")


def test_texture_replace_generates_masked_change() -> None:
    assert_canary_output("texture_replace")


def test_building_crop_add_generates_masked_change(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.npz"
    crop = np.full((4, 4, 3), 0.9, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    crops = np.empty(1, dtype=object)
    masks = np.empty(1, dtype=object)
    crops[0] = crop
    masks[0] = mask
    np.savez_compressed(
        bank_path,
        crops=crops,
        masks=masks,
        metadata=np.asarray(['{"pair_id": "source", "area": 4}']),
    )

    modified_pair, canary_mask = BuildingCropAdd(bank_path=bank_path).generate(
        make_pair(), np.random.default_rng(13)
    )

    assert bool(canary_mask.any())
    assert not np.array_equal(modified_pair.after, make_pair().after)


def test_building_remove_fill_generates_masked_change() -> None:
    assert_canary_output("building_remove_fill")


def test_canary_generation_is_deterministic_for_seeded_rng() -> None:
    first_pair, first_mask = SmallBuildingAdd(size=4).generate(
        make_pair(), np.random.default_rng(11)
    )
    second_pair, second_mask = SmallBuildingAdd(size=4).generate(
        make_pair(), np.random.default_rng(11)
    )

    np.testing.assert_array_equal(first_pair.after, second_pair.after)
    np.testing.assert_array_equal(first_mask, second_mask)


def test_building_crop_add_is_deterministic_for_seeded_rng(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.npz"
    crop = np.full((4, 4, 3), 0.75, dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)
    crops = np.empty(1, dtype=object)
    masks = np.empty(1, dtype=object)
    crops[0] = crop
    masks[0] = mask
    np.savez_compressed(
        bank_path,
        crops=crops,
        masks=masks,
        metadata=np.asarray(['{"pair_id": "source", "area": 16}']),
    )

    generator = BuildingCropAdd(bank_path=bank_path)
    first_pair, first_mask = generator.generate(make_pair(), np.random.default_rng(11))
    second_pair, second_mask = generator.generate(make_pair(), np.random.default_rng(11))

    np.testing.assert_array_equal(first_pair.after, second_pair.after)
    np.testing.assert_array_equal(first_mask, second_mask)


def test_histogram_matching_moves_channel_means_toward_reference() -> None:
    crop = np.zeros((4, 4, 2), dtype=np.float32)
    crop[..., 0] = np.linspace(0.0, 0.3, 16).reshape(4, 4)
    crop[..., 1] = np.linspace(0.7, 1.0, 16).reshape(4, 4)
    reference = np.zeros((4, 4, 2), dtype=np.float32)
    reference[..., 0] = np.linspace(0.6, 0.9, 16).reshape(4, 4)
    reference[..., 1] = np.linspace(0.1, 0.4, 16).reshape(4, 4)

    matched = histogram_match_channels(crop, reference)

    assert abs(float(matched[..., 0].mean()) - float(reference[..., 0].mean())) < abs(
        float(crop[..., 0].mean()) - float(reference[..., 0].mean())
    )
    assert abs(float(matched[..., 1].mean()) - float(reference[..., 1].mean())) < abs(
        float(crop[..., 1].mean()) - float(reference[..., 1].mean())
    )


def test_building_remove_fill_uses_non_constant_non_black_texture() -> None:
    image = np.zeros((24, 24, 3), dtype=np.float32)
    row_values = np.linspace(0.2, 0.8, 24, dtype=np.float32)
    image[:, :, 0] = row_values[:, None]
    image[:, :, 1] = row_values[None, :]
    image[:, :, 2] = 0.4
    image[6:18, 6:18] = 0.95
    pair = ImagePair(before=image.copy(), after=image.copy())

    modified_pair, mask = BuildingRemoveFill(size=8, ring_width=3).generate(
        pair, np.random.default_rng(3)
    )
    filled_pixels = modified_pair.after[mask]

    assert float(filled_pixels.mean()) > 0.0
    assert float(filled_pixels.std()) > 0.0


def test_building_crop_add_requires_bank_path() -> None:
    with pytest.raises(ConfigError, match="canary.bank_path"):
        BuildingCropAdd(bank_path=None)


def test_building_crop_add_rejects_invalid_radiometry_blend(tmp_path: Path) -> None:
    """The configurable alignment factor must remain a convex blend."""
    bank_path = tmp_path / "bank.npz"
    np.savez_compressed(
        bank_path,
        crops=np.empty(0, dtype=object),
        masks=np.empty(0, dtype=object),
        metadata=np.empty(0, dtype=object),
    )

    with pytest.raises(ConfigError, match="radiometry_blend"):
        BuildingCropAdd(bank_path=bank_path, radiometry_blend=1.1)


def test_building_crop_add_stays_visible_against_flat_background(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.npz"
    crop = np.linspace(0.7, 0.9, 48, dtype=np.float32).reshape(4, 4, 3)
    mask = np.ones((4, 4), dtype=bool)
    crops = np.empty(1, dtype=object)
    masks = np.empty(1, dtype=object)
    crops[0] = crop
    masks[0] = mask
    np.savez_compressed(
        bank_path,
        crops=crops,
        masks=masks,
        metadata=np.asarray(['{"pair_id": "source", "area": 16}']),
    )
    flat = np.full((16, 16, 3), 0.1, dtype=np.float32)
    pair = ImagePair(before=flat.copy(), after=flat.copy())

    modified_pair, canary_mask = BuildingCropAdd(bank_path=bank_path).generate(
        pair, np.random.default_rng(5)
    )

    in_mask_delta = np.abs(modified_pair.after - pair.after)[canary_mask].mean()
    assert float(in_mask_delta) >= MIN_VISIBLE_CANARY_DELTA


def test_align_crop_radiometry_preserves_structure() -> None:
    crop = np.linspace(0.5, 1.0, 32, dtype=np.float32).reshape(4, 4, 2)
    reference = np.full((8, 8, 2), 0.2, dtype=np.float32)

    aligned = align_crop_radiometry(crop, reference)

    for channel in range(2):
        original = crop[..., channel].ravel()
        adjusted = aligned[..., channel].ravel()
        correlation = np.corrcoef(original, adjusted)[0, 1]
        assert correlation > 0.99
        assert abs(float(adjusted.mean()) - 0.2) < abs(float(original.mean()) - 0.2)


def test_building_remove_fill_targets_salient_structure() -> None:
    rng_image = np.random.default_rng(0)
    image = np.full((24, 24, 3), 0.2, dtype=np.float32)
    image[4:12, 4:12] = rng_image.random((8, 8, 3), dtype=np.float32)
    structure = np.zeros((24, 24), dtype=bool)
    structure[4:12, 4:12] = True
    pair = ImagePair(before=image.copy(), after=image.copy())

    _, mask = BuildingRemoveFill(size=8, ring_width=3).generate(
        pair, np.random.default_rng(9)
    )

    assert bool((mask & structure).any())
