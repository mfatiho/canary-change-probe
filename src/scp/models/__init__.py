"""Change-detection model adapter interfaces and registry."""

from scp.models.adapters import (
    BITAdapter,
    CDMambaAdapter,
    ChangeFormerAdapter,
    ChangeMambaAdapter,
    TinyCDAdapter,
)
from scp.models.base import CDModelAdapter, ModelLoadError
from scp.models.registry import ModelRegistry, get_model_adapter

__all__ = [
    "BITAdapter",
    "CDModelAdapter",
    "CDMambaAdapter",
    "ChangeFormerAdapter",
    "ChangeMambaAdapter",
    "ModelLoadError",
    "ModelRegistry",
    "TinyCDAdapter",
    "get_model_adapter",
]
