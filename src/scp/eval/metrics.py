"""Offline evaluation metrics that may use ground truth."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

GT_CHANGE_FAILURE_DENSITY_THRESHOLD = 0.01
EMPTY_PREDICTION_DENSITY_THRESHOLD = 0.001
LOW_RESPONSE_FN_RATE_THRESHOLD = 0.5

EvalLabelValue: TypeAlias = str | float | int


@dataclass(frozen=True)
class CounterfactualCanaryMetrics:
    """Canary-mask response before and after a synthetic intervention.

    The original-pair prediction is the counterfactual baseline. Activation
    fractions are normalized by the number of pixels in the inserted canary
    mask, so every field is comparable across canary sizes.
    """

    original_recall: float
    post_recall: float
    recall_delta: float
    original_mean_response: float
    post_mean_response: float
    mean_response_delta: float
    newly_activated_fraction: float
    newly_deactivated_fraction: float


def counterfactual_canary_metrics(
    original_prediction: NDArray[np.floating],
    post_prediction: NDArray[np.floating],
    truth: NDArray[np.bool_],
    threshold: float,
) -> CounterfactualCanaryMetrics:
    """Compare a canary intervention with its original-pair prediction.

    Args:
        original_prediction: Probability map for the unmodified image pair.
        post_prediction: Probability map after inserting the canary.
        truth: Binary mask of pixels modified by the canary generator.
        threshold: Probability threshold used to define active pixels.

    Returns:
        Absolute pre/post responses, post-minus-pre deltas, and activation
        transition fractions within the canary mask.

    Raises:
        ValueError: If arrays are malformed, non-finite, or the canary mask is
            empty.
    """
    original_array = np.asarray(original_prediction, dtype=np.float32)
    post_array = np.asarray(post_prediction, dtype=np.float32)
    truth_mask = np.asarray(truth, dtype=bool)
    _validate_counterfactual_inputs(original_array, post_array, truth_mask, threshold)

    original_inside = original_array[truth_mask]
    post_inside = post_array[truth_mask]
    original_active = original_inside >= threshold
    post_active = post_inside >= threshold

    original_recall = float(np.mean(original_active))
    post_recall = float(np.mean(post_active))
    original_mean_response = float(np.mean(original_inside))
    post_mean_response = float(np.mean(post_inside))
    return CounterfactualCanaryMetrics(
        original_recall=original_recall,
        post_recall=post_recall,
        recall_delta=post_recall - original_recall,
        original_mean_response=original_mean_response,
        post_mean_response=post_mean_response,
        mean_response_delta=post_mean_response - original_mean_response,
        newly_activated_fraction=float(np.mean(~original_active & post_active)),
        newly_deactivated_fraction=float(np.mean(original_active & ~post_active)),
    )


def canary_recall(prediction: NDArray[np.bool_], truth: NDArray[np.bool_]) -> float:
    """Return recall over the known canary mask."""
    prediction_mask, truth_mask = _validate_masks(prediction, truth)
    positives = int(np.sum(truth_mask))
    if positives == 0:
        return 1.0
    return float(np.sum(prediction_mask & truth_mask) / positives)


def canary_precision(prediction: NDArray[np.bool_], truth: NDArray[np.bool_]) -> float:
    """Return precision over the known canary mask."""
    prediction_mask, truth_mask = _validate_masks(prediction, truth)
    predicted = int(np.sum(prediction_mask))
    if predicted == 0:
        return 1.0
    return float(np.sum(prediction_mask & truth_mask) / predicted)


def boundary_iou(
    prediction: NDArray[np.bool_], truth: NDArray[np.bool_], dilation: int = 1
) -> float:
    """Return intersection-over-union of dilated object boundaries."""
    prediction_mask, truth_mask = _validate_masks(prediction, truth)
    prediction_boundary = _boundary(prediction_mask, dilation)
    truth_boundary = _boundary(truth_mask, dilation)
    union = np.sum(prediction_boundary | truth_boundary)
    if union == 0:
        return 1.0
    return float(np.sum(prediction_boundary & truth_boundary) / union)


def auroc(labels: Sequence[int] | NDArray[np.integer], scores: Sequence[float]) -> float:
    """Return area under the ROC curve."""
    return float(roc_auc_score(labels, scores))


def auprc(labels: Sequence[int] | NDArray[np.integer], scores: Sequence[float]) -> float:
    """Return area under the precision-recall curve."""
    return float(average_precision_score(labels, scores))


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Spearman rank correlation."""
    correlation = spearmanr(left, right).correlation
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def false_negative_rate(prediction: NDArray[np.bool_], truth: NDArray[np.bool_]) -> float:
    """Return the fraction of true change pixels missed by a predicted mask."""
    prediction_mask, truth_mask = _validate_masks(prediction, truth)
    positives = int(np.sum(truth_mask))
    if positives == 0:
        return 0.0
    false_negatives = np.sum(truth_mask & ~prediction_mask)
    return float(false_negatives / positives)


def compute_patch_eval_label(
    pair_id: str,
    prediction: NDArray[np.floating],
    truth: NDArray[np.bool_] | NDArray[np.integer],
    threshold: float,
) -> dict[str, EvalLabelValue]:
    """Compute offline silent-failure labels for one predicted patch.

    Args:
        pair_id: Stable image-pair or patch identifier.
        prediction: Model probability map for the pair.
        truth: Binary ground-truth change mask.
        threshold: Probability threshold used to binarize predictions.

    Returns:
        CSV-ready evaluation metrics and binary failure labels.

    Raises:
        ValueError: If prediction and truth shapes differ.
    """
    prediction_array = np.asarray(prediction, dtype=np.float32)
    truth_mask = np.asarray(truth, dtype=bool)
    if prediction_array.shape != truth_mask.shape:
        raise ValueError("prediction and truth must have the same shape")

    prediction_mask = prediction_array >= threshold
    gt_change_density = float(np.mean(truth_mask))
    pred_change_density = float(np.mean(prediction_mask))
    fn_rate = false_negative_rate(prediction_mask, truth_mask)
    has_gt_change = gt_change_density > GT_CHANGE_FAILURE_DENSITY_THRESHOLD
    empty_mask_failure = int(
        has_gt_change and pred_change_density < EMPTY_PREDICTION_DENSITY_THRESHOLD
    )
    low_response_failure = int(has_gt_change and fn_rate > LOW_RESPONSE_FN_RATE_THRESHOLD)
    return {
        "pair_id": pair_id,
        "gt_change_density": gt_change_density,
        "pred_change_density": pred_change_density,
        "fn_rate": fn_rate,
        "empty_mask_failure": empty_mask_failure,
        "low_response_failure": low_response_failure,
    }


def image_silent_failure_labels(
    gt_masks: NDArray[np.bool_],
    predictions: NDArray[np.floating],
    response_threshold: float,
) -> NDArray[np.int_]:
    """Label images as silent failures when GT has change but response is silent."""
    if gt_masks.shape != predictions.shape:
        raise ValueError("gt_masks and predictions must have the same shape")
    if gt_masks.ndim < 3:
        raise ValueError("gt_masks and predictions must be image batches")
    has_change = np.any(gt_masks.astype(bool), axis=tuple(range(1, gt_masks.ndim)))
    has_response = np.any(predictions >= response_threshold, axis=tuple(range(1, predictions.ndim)))
    return (has_change & ~has_response).astype(int)


def _validate_masks(
    prediction: NDArray[np.bool_], truth: NDArray[np.bool_]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    prediction_mask = np.asarray(prediction, dtype=bool)
    truth_mask = np.asarray(truth, dtype=bool)
    if prediction_mask.shape != truth_mask.shape:
        raise ValueError("prediction and truth masks must have the same shape")
    return prediction_mask, truth_mask


def _validate_counterfactual_inputs(
    original_prediction: NDArray[np.float32],
    post_prediction: NDArray[np.float32],
    truth_mask: NDArray[np.bool_],
    threshold: float,
) -> None:
    if original_prediction.shape != truth_mask.shape:
        raise ValueError("original prediction and canary mask must have the same shape")
    if post_prediction.shape != truth_mask.shape:
        raise ValueError("post-intervention prediction and canary mask must have the same shape")
    if not np.any(truth_mask):
        raise ValueError("canary mask must be non-empty")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not np.all(np.isfinite(original_prediction)):
        raise ValueError("original prediction must contain only finite values")
    if not np.all(np.isfinite(post_prediction)):
        raise ValueError("post-intervention prediction must contain only finite values")


def _boundary(mask: NDArray[np.bool_], dilation: int) -> NDArray[np.bool_]:
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    boundary = mask & ~eroded
    if dilation <= 0:
        return boundary
    return binary_dilation(boundary, iterations=dilation)
