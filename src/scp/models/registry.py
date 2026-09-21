"""Model adapter registry."""

from __future__ import annotations

from collections.abc import Callable

from scp.models.adapters import (
    BITAdapter,
    CDMambaAdapter,
    ChangeFormerAdapter,
    ChangeMambaAdapter,
    TinyCDAdapter,
)
from scp.models.base import CDModelAdapter

ModelFactory = Callable[..., CDModelAdapter]


class ModelRegistry:
    """Register and instantiate model adapters by name."""

    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(self, name: str, factory: ModelFactory) -> None:
        """Register a model adapter factory."""
        if not name:
            raise ValueError("model name must be non-empty")
        if name in self._factories:
            raise KeyError(f"model already registered: {name}")
        self._factories[name] = factory

    def get(self, name: str) -> ModelFactory:
        """Return the model adapter factory for a registered name."""
        try:
            return self._factories[name]
        except KeyError as error:
            raise KeyError(f"unknown model adapter: {name}") from error

    def create(self, name: str, *args: object, **kwargs: object) -> CDModelAdapter:
        """Instantiate a registered model adapter."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered model adapter names in sorted order."""
        return tuple(sorted(self._factories))


MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register("bit", BITAdapter)
MODEL_REGISTRY.register("cdmamba", CDMambaAdapter)
MODEL_REGISTRY.register("changeformer", ChangeFormerAdapter)
MODEL_REGISTRY.register("changemamba", ChangeMambaAdapter)
MODEL_REGISTRY.register("tinycd", TinyCDAdapter)


def get_model_adapter(name: str, *args: object, **kwargs: object) -> CDModelAdapter:
    """Create a model adapter from the global registry."""
    return MODEL_REGISTRY.create(name, *args, **kwargs)
