"""Synthetic canary change generators."""

from scp.canary.base import CanaryGenerator
from scp.canary.generators import (
    BuildingCropAdd,
    BuildingRemove,
    BuildingRemoveFill,
    LinearRoadAdd,
    SmallBuildingAdd,
    TextureReplace,
)
from scp.canary.registry import CANARY_REGISTRY, CanaryRegistry, get_canary_generator

__all__ = [
    "BuildingRemove",
    "BuildingCropAdd",
    "BuildingRemoveFill",
    "CANARY_REGISTRY",
    "CanaryGenerator",
    "CanaryRegistry",
    "LinearRoadAdd",
    "SmallBuildingAdd",
    "TextureReplace",
    "get_canary_generator",
]
