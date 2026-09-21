"""Source-calibrated z-score normalization."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

MIN_STANDARD_DEVIATION = 1e-12


@dataclass(frozen=True)
class ZScoreNormalizer:
    """Normalizes metrics using means and standard deviations from source calibration."""

    means: dict[str, float]
    standard_deviations: dict[str, float]

    @classmethod
    def fit(cls, samples: Mapping[str, Sequence[float]]) -> "ZScoreNormalizer":
        """Fit z-normalization statistics from source calibration samples."""
        if not samples:
            raise ValueError("samples must contain at least one metric")
        means: dict[str, float] = {}
        standard_deviations: dict[str, float] = {}
        for name, values in samples.items():
            array = np.asarray(values, dtype=np.float64)
            if array.size == 0:
                raise ValueError(f"metric '{name}' has no calibration values")
            means[name] = float(np.mean(array))
            standard_deviations[name] = max(float(np.std(array)), MIN_STANDARD_DEVIATION)
        return cls(means=means, standard_deviations=standard_deviations)

    def transform(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """Transform metrics to z-scores using fitted source statistics."""
        normalized: dict[str, float] = {}
        for name, mean in self.means.items():
            if name not in metrics:
                raise KeyError(f"missing metric for normalization: {name}")
            normalized[name] = (float(metrics[name]) - mean) / self.standard_deviations[name]
        return normalized

