"""Ground-truth-based evaluation metrics for offline analysis only."""

from scp.eval.metrics import (
    CounterfactualCanaryMetrics,
    EMPTY_PREDICTION_DENSITY_THRESHOLD,
    GT_CHANGE_FAILURE_DENSITY_THRESHOLD,
    LOW_RESPONSE_FN_RATE_THRESHOLD,
    auprc,
    auroc,
    boundary_iou,
    canary_precision,
    canary_recall,
    compute_patch_eval_label,
    counterfactual_canary_metrics,
    false_negative_rate,
    image_silent_failure_labels,
    spearman,
)

__all__ = [
    "CounterfactualCanaryMetrics",
    "EMPTY_PREDICTION_DENSITY_THRESHOLD",
    "GT_CHANGE_FAILURE_DENSITY_THRESHOLD",
    "LOW_RESPONSE_FN_RATE_THRESHOLD",
    "auprc",
    "auroc",
    "boundary_iou",
    "canary_precision",
    "canary_recall",
    "compute_patch_eval_label",
    "counterfactual_canary_metrics",
    "false_negative_rate",
    "image_silent_failure_labels",
    "spearman",
]
