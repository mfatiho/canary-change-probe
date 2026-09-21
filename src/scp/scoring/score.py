"""Scalar risk score components."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class PassiveScorer:
    """Weighted aggregation of passive signal z-scores."""

    weights: Mapping[str, float]

    def score(self, metrics: Mapping[str, float]) -> float:
        """Compute a weighted passive score."""
        if not self.weights:
            raise ValueError("weights must not be empty")
        score = 0.0
        for name, weight in self.weights.items():
            if name not in metrics:
                raise KeyError(f"missing passive metric: {name}")
            score += float(weight) * float(metrics[name])
        return score


@dataclass(frozen=True)
class MismatchScorer:
    """Compares image evidence against model response to expose silent mismatch."""

    image_evidence_key: str
    response_key: str

    def score(self, metrics: Mapping[str, float]) -> float:
        """Compute positive mismatch when image evidence exceeds model response."""
        if self.image_evidence_key not in metrics:
            raise KeyError(f"missing image evidence metric: {self.image_evidence_key}")
        if self.response_key not in metrics:
            raise KeyError(f"missing response metric: {self.response_key}")
        return float(metrics[self.image_evidence_key]) - float(metrics[self.response_key])


@dataclass(frozen=True)
class CombinedScorer:
    """Combines passive and mismatch scores with injected weights."""

    passive_weight: float
    mismatch_weight: float

    def score(self, passive_score: float, mismatch_score: float) -> float:
        """Return the combined silent-failure risk score."""
        return self.passive_weight * passive_score + self.mismatch_weight * mismatch_score

