import numpy as np
import pytest

from scp.data import ImagePair
from scp.signals import (
    CVAImageDifference,
    EvidenceCoverage,
    LowThresholdResponse,
    MaxConfidence,
    NoChangeSaturation,
    PredictionDensity,
    PredictionEntropy,
    ResponsivenessCorrelation,
    TemporalSwapConsistency,
    get_signal,
)


def make_pair() -> ImagePair:
    before = np.zeros((3, 3), dtype=np.float32)
    after = before.copy()
    after[1, 1] = 1.0
    return ImagePair(before=before, after=after)


def test_prediction_density_counts_high_confidence_pixels() -> None:
    prediction = np.array([[0.1, 0.9], [0.7, 0.2]], dtype=np.float32)
    pair = ImagePair(before=np.zeros((2, 2), dtype=np.float32), after=np.ones((2, 2)))

    assert PredictionDensity(threshold=0.5).compute(pair, prediction) == pytest.approx(
        0.5
    )


def test_low_threshold_response_counts_weak_predictions() -> None:
    prediction = np.array([[0.1, 0.4], [0.7, 0.2]], dtype=np.float32)
    pair = ImagePair(before=np.zeros((2, 2), dtype=np.float32), after=np.ones((2, 2)))

    assert LowThresholdResponse(low_threshold=0.3).compute(pair, prediction) == pytest.approx(
        0.5
    )


def test_cva_image_difference_uses_image_evidence() -> None:
    assert CVAImageDifference().compute(make_pair(), np.zeros((3, 3))) == pytest.approx(1.0 / 9.0)


def test_responsiveness_correlation_tracks_image_evidence() -> None:
    pair = make_pair()
    prediction = np.zeros((3, 3), dtype=np.float32)
    prediction[1, 1] = 1.0

    assert ResponsivenessCorrelation().compute(pair, prediction) == pytest.approx(1.0)


def test_temporal_swap_consistency_is_one_for_identical_maps() -> None:
    prediction = np.full((3, 3), 0.2, dtype=np.float32)

    assert TemporalSwapConsistency().compute(
        make_pair(), prediction, swapped_prediction=prediction.copy()
    ) == pytest.approx(1.0)


def test_no_change_saturation_counts_low_response_pixels() -> None:
    prediction = np.array([[0.1, 0.9], [0.4, 0.2]], dtype=np.float32)
    pair = ImagePair(before=np.zeros((2, 2), dtype=np.float32), after=np.ones((2, 2)))

    assert NoChangeSaturation(silence_threshold=0.3).compute(pair, prediction) == pytest.approx(
        0.5
    )


def test_prediction_entropy_is_zero_at_binary_extremes_and_maximal_at_half() -> None:
    pair = ImagePair(before=np.zeros((2, 2), dtype=np.float32), after=np.ones((2, 2)))
    binary_prediction = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    half_prediction = np.full((2, 2), 0.5, dtype=np.float32)

    entropy = PredictionEntropy()

    assert entropy.compute(pair, binary_prediction) == pytest.approx(0.0, abs=1e-5)
    assert entropy.compute(pair, half_prediction) == pytest.approx(np.log(2.0))


def test_max_confidence_is_one_at_binary_extremes_and_half_at_uncertain_half() -> None:
    pair = ImagePair(before=np.zeros((2, 2), dtype=np.float32), after=np.ones((2, 2)))
    binary_prediction = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    half_prediction = np.full((2, 2), 0.5, dtype=np.float32)

    max_confidence = MaxConfidence()

    assert max_confidence.compute(pair, binary_prediction) == pytest.approx(1.0)
    assert max_confidence.compute(pair, half_prediction) == pytest.approx(0.5)


def test_signal_registry_exposes_builtin_signal() -> None:
    assert isinstance(get_signal("prediction_density"), PredictionDensity)
    assert isinstance(get_signal("prediction_entropy"), PredictionEntropy)
    assert isinstance(get_signal("max_confidence"), MaxConfidence)


def _evidence_pair(evidence_pixels: list[tuple[int, int]]) -> ImagePair:
    before = np.zeros((10, 10), dtype=np.float32)
    after = before.copy()
    for row, col in evidence_pixels:
        after[row, col] = 1.0
    return ImagePair(before=before, after=after)


def test_evidence_coverage_full_when_model_responds_on_evidence() -> None:
    pair = _evidence_pair([(2, 2), (7, 7)])
    prediction = np.zeros((10, 10), dtype=np.float32)
    prediction[2, 2] = prediction[7, 7] = 0.8

    signal = EvidenceCoverage(evidence_percentile=90.0, response_threshold=0.1)
    assert signal.compute(pair, prediction) == pytest.approx(1.0)


def test_evidence_coverage_zero_for_silent_model() -> None:
    pair = _evidence_pair([(2, 2), (7, 7)])
    prediction = np.zeros((10, 10), dtype=np.float32)

    signal = EvidenceCoverage(evidence_percentile=90.0, response_threshold=0.1)
    assert signal.compute(pair, prediction) == pytest.approx(0.0)


def test_evidence_coverage_invariant_to_scene_change_density() -> None:
    sparse_pixels = [(2, 2)]
    dense_pixels = [(r, c) for r in range(3, 8) for c in range(3, 8)]
    signal = EvidenceCoverage(evidence_percentile=90.0, response_threshold=0.1)

    coverages = []
    for pixels in (sparse_pixels, dense_pixels):
        pair = _evidence_pair(pixels)
        prediction = np.zeros((10, 10), dtype=np.float32)
        for row, col in pixels:
            prediction[row, col] = 0.9
        coverages.append(signal.compute(pair, prediction))

    assert coverages[0] == pytest.approx(coverages[1])
    assert coverages[0] == pytest.approx(1.0)
