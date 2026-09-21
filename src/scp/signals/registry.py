"""Signal registry."""

from __future__ import annotations

from collections.abc import Callable

from scp.signals.base import PassiveSignal
from scp.signals.implementations import (
    CVAImageDifference,
    EvidenceCoverage,
    LowThresholdResponse,
    MaxConfidence,
    NoChangeSaturation,
    PredictionDensity,
    PredictionEntropy,
    ResponsivenessCorrelation,
    TemporalSwapConsistency,
)

SignalFactory = Callable[..., PassiveSignal]


class SignalRegistry:
    """Register and create passive signal implementations."""

    def __init__(self) -> None:
        self._factories: dict[str, SignalFactory] = {}

    def register(self, name: str, factory: SignalFactory) -> None:
        """Register a passive signal factory."""
        if not name:
            raise ValueError("signal name must be non-empty")
        if name in self._factories:
            raise KeyError(f"signal already registered: {name}")
        self._factories[name] = factory

    def get(self, name: str) -> SignalFactory:
        """Return a passive signal factory by name."""
        try:
            return self._factories[name]
        except KeyError as error:
            raise KeyError(f"unknown signal: {name}") from error

    def create(self, name: str, *args: object, **kwargs: object) -> PassiveSignal:
        """Create a passive signal by name."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered signal names in registration order."""
        return tuple(self._factories)


SIGNAL_REGISTRY = SignalRegistry()
SIGNAL_REGISTRY.register("prediction_density", PredictionDensity)
SIGNAL_REGISTRY.register("low_threshold_response", LowThresholdResponse)
SIGNAL_REGISTRY.register("cva_image_difference", CVAImageDifference)
SIGNAL_REGISTRY.register("responsiveness_correlation", ResponsivenessCorrelation)
SIGNAL_REGISTRY.register("temporal_swap_consistency", TemporalSwapConsistency)
SIGNAL_REGISTRY.register("no_change_saturation", NoChangeSaturation)
SIGNAL_REGISTRY.register("evidence_coverage", EvidenceCoverage)
SIGNAL_REGISTRY.register("prediction_entropy", PredictionEntropy)
SIGNAL_REGISTRY.register("max_confidence", MaxConfidence)


def get_signal(name: str, *args: object, **kwargs: object) -> PassiveSignal:
    """Create a passive signal from the global registry."""
    return SIGNAL_REGISTRY.create(name, *args, **kwargs)
