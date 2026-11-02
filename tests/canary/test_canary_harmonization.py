"""Opt-in harmonization of the synthetic road and texture canaries.

Defaults must reproduce the frozen primary probes bit-for-bit; the opt-in
variants only feather the boundary outside the scored mask or match the canary
radiometry to the target after-image.
"""

import hashlib

import numpy as np
import pytest

from scp.canary.generators import LinearRoadAdd, TextureReplace
from scp.data import ImagePair
from scp.utils import ConfigError

FROZEN_DIGESTS = {
    ("LinearRoadAdd", 0): "3fa6521f2ed42a85",
    ("LinearRoadAdd", 1): "d983c445094b9201",
    ("LinearRoadAdd", 2): "ed3a2c671524f0fd",
    ("TextureReplace", 0): "30122f7b930f9298",
    ("TextureReplace", 1): "ef87f9142b0e628e",
    ("TextureReplace", 2): "e8d203cb5a8814d5",
}
"""SHA-256 prefixes recorded from the primary generators before the opt-in
parameters existed."""

FEATHER_PX = 4


def make_uint8_pair() -> ImagePair:
    image = np.random.default_rng(3).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return ImagePair(before=image.copy(), after=image.copy())


def digest(pair: ImagePair, mask: np.ndarray) -> str:
    return hashlib.sha256(pair.after.tobytes() + mask.tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("generator", [LinearRoadAdd(), TextureReplace()])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_generators_reproduce_frozen_probes(generator, seed: int) -> None:
    modified, mask = generator.generate(make_uint8_pair(), np.random.default_rng(seed))

    assert digest(modified, mask) == FROZEN_DIGESTS[(type(generator).__name__, seed)]


@pytest.mark.parametrize("generator_type", [LinearRoadAdd, TextureReplace])
def test_feathering_keeps_scored_mask_and_placement(generator_type) -> None:
    pair = make_uint8_pair()
    _, hard_mask = generator_type().generate(pair, np.random.default_rng(5))
    feathered, soft_mask = generator_type(edge_feather_px=FEATHER_PX).generate(
        pair, np.random.default_rng(5)
    )

    np.testing.assert_array_equal(soft_mask, hard_mask)
    changed = np.any(feathered.after != pair.after, axis=-1)
    assert np.any(changed & ~hard_mask), "feathering must blend a band outside the mask"


def test_feathered_road_core_is_unchanged_and_band_is_intermediate() -> None:
    pair = make_uint8_pair()
    hard, mask = LinearRoadAdd().generate(pair, np.random.default_rng(5))
    soft, _ = LinearRoadAdd(edge_feather_px=FEATHER_PX).generate(pair, np.random.default_rng(5))

    np.testing.assert_array_equal(soft.after[mask], hard.after[mask])
    band = np.any(soft.after != pair.after, axis=-1) & ~mask
    road_value = hard.after[mask][0].astype(np.int16)
    original = pair.after[band].astype(np.int16)
    blended = soft.after[band].astype(np.int16)
    assert np.all(np.abs(blended - road_value) <= np.abs(original - road_value))


def test_feathering_leaves_pixels_beyond_band_unchanged() -> None:
    pair = make_uint8_pair()
    soft, mask = TextureReplace(edge_feather_px=FEATHER_PX).generate(
        pair, np.random.default_rng(9)
    )
    rows, cols = np.nonzero(mask)
    outside = np.ones(mask.shape, dtype=bool)
    outside[
        max(0, rows.min() - FEATHER_PX) : rows.max() + FEATHER_PX + 1,
        max(0, cols.min() - FEATHER_PX) : cols.max() + FEATHER_PX + 1,
    ] = False

    np.testing.assert_array_equal(soft.after[outside], pair.after[outside])


def test_scene_matched_road_uses_after_image_quantile() -> None:
    pair = make_uint8_pair()
    modified, mask = LinearRoadAdd(scene_radiometry_blend=1.0).generate(
        pair, np.random.default_rng(5)
    )
    expected = np.quantile(pair.after.reshape(-1, 3).astype(np.float64), 0.8, axis=0)

    np.testing.assert_allclose(modified.after[mask][0], np.rint(expected), atol=1)


def test_scene_matched_texture_follows_after_image_histogram() -> None:
    image = np.random.default_rng(4).integers(40, 90, (64, 64, 3), dtype=np.uint8)
    pair = ImagePair(before=image.copy(), after=image.copy())
    modified, mask = TextureReplace(scene_radiometry_blend=1.0).generate(
        pair, np.random.default_rng(5)
    )
    inserted = modified.after[mask]

    assert inserted.min() >= 40 and inserted.max() < 90
    assert inserted.std() > 0, "matched texture must remain a textured change"


@pytest.mark.parametrize("generator_type", [LinearRoadAdd, TextureReplace])
@pytest.mark.parametrize(
    "params",
    [{"edge_feather_px": -1}, {"scene_radiometry_blend": -0.1}, {"scene_radiometry_blend": 1.5}],
)
def test_invalid_harmonization_parameters_fail(generator_type, params) -> None:
    with pytest.raises(ConfigError):
        generator_type(**params)
