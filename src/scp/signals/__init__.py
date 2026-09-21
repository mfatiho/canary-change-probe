"""Passive responsiveness signals."""

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
from scp.signals.registry import SIGNAL_REGISTRY, SignalRegistry, get_signal

__all__ = [
    "CVAImageDifference",
    "EvidenceCoverage",
    "LowThresholdResponse",
    "MaxConfidence",
    "NoChangeSaturation",
    "PassiveSignal",
    "PredictionDensity",
    "PredictionEntropy",
    "ResponsivenessCorrelation",
    "SIGNAL_REGISTRY",
    "SignalRegistry",
    "TemporalSwapConsistency",
    "get_signal",
]
