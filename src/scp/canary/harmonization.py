"""Opt-in boundary and radiometric harmonization for synthetic canaries.

The primary road and texture probes are hard-edged and use fixed intensities.
These helpers support a post-hoc sensitivity check of that choice: feathering
blends the canary into a band *outside* the scored mask, so the scored region is
unchanged, and scene matching moves the canary radiometry toward the target
after-image's own value distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from scp.utils import ConfigError

FloatArray = NDArray[np.float64]


def validate_harmonization(
    owner: str, edge_feather_px: int, scene_radiometry_blend: float
) -> None:
    """Reject harmonization settings outside their defined ranges.

    Raises:
        ConfigError: If the feather width is negative or non-integral, or the
            radiometry blend lies outside [0, 1].
    """
    if edge_feather_px < 0 or int(edge_feather_px) != edge_feather_px:
        raise ConfigError(f"{owner}.edge_feather_px must be a non-negative integer")
    if not 0.0 <= scene_radiometry_blend <= 1.0:
        raise ConfigError(f"{owner}.scene_radiometry_blend must be in [0, 1]")


def is_harmonized(edge_feather_px: int, scene_radiometry_blend: float) -> bool:
    """Return whether any opt-in harmonization is active."""
    return edge_feather_px > 0 or scene_radiometry_blend > 0.0


def feather_alpha(mask: NDArray[np.bool_], feather_px: int) -> FloatArray:
    """Return opacity 1 inside `mask`, decaying linearly to 0 over `feather_px` pixels."""
    if feather_px == 0:
        return mask.astype(np.float64)
    distance = distance_transform_edt(~mask)
    return np.clip(1.0 - distance / (feather_px + 1), 0.0, 1.0)


def composite(
    image: NDArray[np.floating] | NDArray[np.integer],
    layer: FloatArray,
    mask: NDArray[np.bool_],
    feather_px: int,
) -> FloatArray:
    """Alpha-composite `layer` over `image`; the scored mask always has opacity 1."""
    alpha = feather_alpha(mask, feather_px)
    if image.ndim == 3:
        alpha = alpha[..., np.newaxis]
    return alpha * layer + (1.0 - alpha) * np.asarray(image, dtype=np.float64)


def scene_quantile(
    image: NDArray[np.floating] | NDArray[np.integer], quantile: float
) -> FloatArray:
    """Return the per-channel `quantile` of the image (a scalar array for grayscale)."""
    values = np.asarray(image, dtype=np.float64)
    if values.ndim == 2:
        return np.asarray(np.quantile(values, quantile))
    return np.quantile(values.reshape(-1, values.shape[-1]), quantile, axis=0)


def match_to_scene(
    texture: FloatArray, image: NDArray[np.floating] | NDArray[np.integer]
) -> FloatArray:
    """Map each texture channel onto the image channel's empirical distribution.

    Unlike `histogram_match_channels`, the texture and image may differ in size:
    texture ranks are mapped to image quantiles.
    """
    reference = np.asarray(image, dtype=np.float64)
    if texture.ndim == 2:
        return _match_channel(texture, reference)
    return np.stack(
        [
            _match_channel(texture[..., channel], reference[..., channel])
            for channel in range(texture.shape[-1])
        ],
        axis=-1,
    )


def _match_channel(texture: FloatArray, reference: FloatArray) -> FloatArray:
    ranks = np.argsort(np.argsort(texture.ravel(), kind="stable"), kind="stable")
    quantiles = (ranks + 0.5) / ranks.size
    return np.quantile(reference.ravel(), quantiles).reshape(texture.shape)
