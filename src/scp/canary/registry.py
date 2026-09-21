"""Canary generator registry."""

from __future__ import annotations

from collections.abc import Callable

from scp.canary.base import CanaryGenerator
from scp.canary.generators import (
    BuildingCropAdd,
    BuildingRemove,
    BuildingRemoveFill,
    LinearRoadAdd,
    SmallBuildingAdd,
    TextureReplace,
)

CanaryFactory = Callable[..., CanaryGenerator]


class CanaryRegistry:
    """Register and instantiate synthetic canary generators."""

    def __init__(self) -> None:
        self._factories: dict[str, CanaryFactory] = {}

    def register(self, name: str, factory: CanaryFactory) -> None:
        """Register a canary generator factory."""
        if not name:
            raise ValueError("canary name must be non-empty")
        if name in self._factories:
            raise KeyError(f"canary already registered: {name}")
        self._factories[name] = factory

    def get(self, name: str) -> CanaryFactory:
        """Return a canary generator factory by name."""
        try:
            return self._factories[name]
        except KeyError as error:
            raise KeyError(f"unknown canary generator: {name}") from error

    def create(self, name: str, *args: object, **kwargs: object) -> CanaryGenerator:
        """Create a canary generator by name."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered canary names in sorted order."""
        return tuple(sorted(self._factories))


CANARY_REGISTRY = CanaryRegistry()
CANARY_REGISTRY.register("small_building_add", SmallBuildingAdd)
CANARY_REGISTRY.register("building_remove", BuildingRemove)
CANARY_REGISTRY.register("building_crop_add", BuildingCropAdd)
CANARY_REGISTRY.register("building_remove_fill", BuildingRemoveFill)
CANARY_REGISTRY.register("linear_road_add", LinearRoadAdd)
CANARY_REGISTRY.register("texture_replace", TextureReplace)


def get_canary_generator(name: str, *args: object, **kwargs: object) -> CanaryGenerator:
    """Create a canary generator from the global registry."""
    return CANARY_REGISTRY.create(name, *args, **kwargs)
