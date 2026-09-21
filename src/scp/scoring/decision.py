"""Decision policy for silent-failure risk scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["deploy", "warn", "reject"]


@dataclass(frozen=True)
class DecisionThresholds:
    """Injected score thresholds for deploy, warn, and reject decisions."""

    deploy: float
    reject: float

    def __post_init__(self) -> None:
        """Validate monotonic decision thresholds."""
        if self.deploy >= self.reject:
            raise ValueError("deploy threshold must be lower than reject threshold")


@dataclass(frozen=True)
class DecisionPolicy:
    """Maps a scalar risk score to an operational decision."""

    thresholds: DecisionThresholds

    def decide(self, score: float) -> Decision:
        """Return deploy, warn, or reject for a risk score."""
        if score <= self.thresholds.deploy:
            return "deploy"
        if score >= self.thresholds.reject:
            return "reject"
        return "warn"

