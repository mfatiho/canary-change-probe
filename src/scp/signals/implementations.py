"""Pure NumPy passive responsiveness signal implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scp.data import ImagePair
from scp.signals.base import PassiveSignal

MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0
ENTROPY_EPS = 1e-7


@dataclass(frozen=True)
class PredictionDensity(PassiveSignal):
    """Fraction of pixels above a high-confidence change threshold."""

    threshold: float = 0.5

    @property
    def name(self) -> str:
        """Signal registry name."""
        return "prediction_density"

    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute the high-confidence predicted-change density."""
        _validate_prediction(pair, prediction)
        return float(np.mean(prediction >= self.threshold))


@dataclass(frozen=True)
class LowThresholdResponse(PassiveSignal):
    """Fraction of pixels with any weak model response."""

    low_threshold: float = 0.1

    @property
    def name(self) -> str:
        """Signal registry name."""
        return "low_threshold_response"

    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute weak-response density."""
        _validate_prediction(pair, prediction)
        return float(np.mean(prediction >= self.low_threshold))


class CVAImageDifference(PassiveSignal):
    """Mean change-vector-analysis evidence from image differences."""

    @property
    def name(self) -> str:
        """Signal registry name."""
        return "cva_image_difference"

    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute mean absolute image-difference evidence."""
        _validate_prediction(pair, prediction)
        return float(np.mean(cva_evidence_map(pair)))


class ResponsivenessCorrelation(PassiveSignal):
    """Pearson correlation between image evidence and model response."""

    @property
    def name(self) -> str:
        """Signal registry name."""
        return "responsiveness_correlation"

    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute correlation between CVA evidence and predicted response."""
        _validate_prediction(pair, prediction)
        evidence = cva_evidence_map(pair).ravel()
        response = np.asarray(prediction, dtype=np.float32).ravel()
        if np.all(evidence == evidence[0]) or np.all(response == response[0]):
            return 0.0
        return float(np.corrcoef(evidence, response)[0, 1])


class TemporalSwapConsistency(PassiveSignal):
    """Agreement between original and temporally swapped predictions."""

    @property
    def name(self) -> str:
        """Signal registry name."""
        return "temporal_swap_consistency"

    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute one minus mean absolute difference between swap predictions."""
        _validate_prediction(pair, prediction)
        if swapped_prediction is None:
            raise ValueError("swapped_prediction is required for temporal swap consistency")
        _validate_prediction(pair, swapped_prediction)
        difference = np.mean(np.abs(prediction - swapped_prediction))
        return float(np.clip(1.0 - difference, MIN_PROBABILITY, MAX_PROBABILITY))


@dataclass(frozen=True)
class NoChangeSaturation(PassiveSignal):
    """Fraction of pixels whose model response is effectively silent."""

    silence_threshold: float = 0.05

    @property
    def name(self) -> str:
        """Signal registry name."""
        return "no_change_saturation"

    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute silent-pixel fraction."""
        _validate_prediction(pair, prediction)
        return float(np.mean(prediction <= self.silence_threshold))


@dataclass(frozen=True)
class PredictionEntropy(PassiveSignal):
    """Mean binary entropy of the predicted change probabilities."""

    eps: float = ENTROPY_EPS

    @property
    def name(self) -> str:
        """Signal registry name."""
        return "prediction_entropy"

    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute mean per-pixel binary entropy with clipped probabilities."""
        _validate_prediction(pair, prediction)
        probabilities = np.clip(prediction.astype(np.float64), self.eps, 1.0 - self.eps)
        entropy = -probabilities * np.log(probabilities)
        entropy -= (1.0 - probabilities) * np.log(1.0 - probabilities)
        return float(np.mean(entropy))


class MaxConfidence(PassiveSignal):
    """Mean max class confidence implied by binary change probabilities."""

    @property
    def name(self) -> str:
        """Signal registry name."""
        return "max_confidence"

    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute the mean of max(p, 1 - p) over pixels."""
        _validate_prediction(pair, prediction)
        probabilities = np.asarray(prediction, dtype=np.float64)
        return float(np.mean(np.maximum(probabilities, 1.0 - probabilities)))


@dataclass(frozen=True)
class EvidenceCoverage(PassiveSignal):
    """Fraction of the pair's strongest-evidence pixels that receive model response.

    Spatially conditioned within the pair: the evidence threshold is the pair's
    own CVA percentile, so the signal is insensitive to how change-dense the scene
    or domain is, unlike domain-mean evidence/response ratios, which inherit
    scene-content confounds.
    """

    evidence_percentile: float = 90.0
    response_threshold: float = 0.1
    evidence_floor: float = 1e-6
    """Keeps the evidence mask on strictly positive evidence when the scene is so
    change-sparse that the percentile itself is zero (else the mask would cover
    the whole image and dilute coverage)."""

    @property
    def name(self) -> str:
        """Signal registry name."""
        return "evidence_coverage"

    def compute(
        self,
        pair: ImagePair,
        prediction: NDArray[np.floating],
        swapped_prediction: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute response coverage over the pair's top-evidence pixels."""
        _validate_prediction(pair, prediction)
        evidence = cva_evidence_map(pair)
        threshold = max(
            float(np.percentile(evidence, self.evidence_percentile)),
            self.evidence_floor,
        )
        evidence_mask = evidence >= threshold
        if not bool(evidence_mask.any()):
            return 0.0
        return float(np.mean(prediction[evidence_mask] >= self.response_threshold))


def cva_evidence_map(pair: ImagePair) -> NDArray[np.float32]:
    """Return per-pixel absolute image-difference evidence."""
    difference = np.abs(pair.after.astype(np.float32) - pair.before.astype(np.float32))
    if difference.ndim == 2:
        return difference.astype(np.float32)
    return np.mean(difference, axis=2).astype(np.float32)


def _validate_prediction(pair: ImagePair, prediction: NDArray[np.floating]) -> None:
    if not isinstance(prediction, np.ndarray):
        raise TypeError("prediction must be a NumPy array")
    if prediction.shape != pair.spatial_shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match {pair.spatial_shape}")
