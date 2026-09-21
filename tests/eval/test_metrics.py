import numpy as np
import pytest

from scp.eval import (
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


def test_canary_recall_and_precision_measure_mask_overlap() -> None:
    truth = np.array([[True, True], [False, False]])
    prediction = np.array([[True, False], [True, False]])

    assert canary_recall(prediction, truth) == pytest.approx(0.5)
    assert canary_precision(prediction, truth) == pytest.approx(0.5)


def test_counterfactual_canary_metrics_isolates_new_and_lost_activations() -> None:
    truth = np.array([[True, True], [True, False]])
    original = np.array([[0.1, 0.8], [0.7, 0.9]], dtype=np.float32)
    post = np.array([[0.9, 0.6], [0.2, 0.1]], dtype=np.float32)

    metrics = counterfactual_canary_metrics(original, post, truth, threshold=0.5)

    assert metrics.original_recall == pytest.approx(2.0 / 3.0)
    assert metrics.post_recall == pytest.approx(2.0 / 3.0)
    assert metrics.recall_delta == pytest.approx(0.0)
    assert metrics.original_mean_response == pytest.approx(1.6 / 3.0)
    assert metrics.post_mean_response == pytest.approx(1.7 / 3.0)
    assert metrics.mean_response_delta == pytest.approx(0.1 / 3.0)
    assert metrics.newly_activated_fraction == pytest.approx(1.0 / 3.0)
    assert metrics.newly_deactivated_fraction == pytest.approx(1.0 / 3.0)


def test_counterfactual_canary_metrics_rejects_empty_mask() -> None:
    prediction = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="must be non-empty"):
        counterfactual_canary_metrics(
            prediction,
            prediction,
            np.zeros((2, 2), dtype=bool),
            threshold=0.5,
        )


def test_boundary_iou_scores_aligned_boundaries() -> None:
    truth = np.zeros((6, 6), dtype=bool)
    prediction = np.zeros((6, 6), dtype=bool)
    truth[2:4, 2:4] = True
    prediction[2:4, 2:4] = True

    assert boundary_iou(prediction, truth) == pytest.approx(1.0)


def test_auc_metrics_use_probabilistic_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert auroc(labels, scores) == pytest.approx(1.0)
    assert auprc(labels, scores) == pytest.approx(1.0)


def test_spearman_measures_rank_correlation() -> None:
    assert spearman([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_false_negative_rate_counts_missed_change_pixels() -> None:
    truth = np.array([[True, True], [False, False]])
    prediction = np.array([[True, False], [False, False]])

    assert false_negative_rate(prediction, truth) == pytest.approx(0.5)


def test_image_silent_failure_labels_use_gt_only_in_eval() -> None:
    gt_masks = np.array(
        [
            [[True, False], [False, False]],
            [[False, False], [False, False]],
        ]
    )
    predictions = np.array(
        [
            [[0.1, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )

    assert image_silent_failure_labels(gt_masks, predictions, response_threshold=0.5).tolist() == [
        1,
        0,
    ]


def test_compute_patch_eval_label_uses_density_and_false_negative_rules() -> None:
    truth = np.array([[True, True], [False, False]])
    prediction = np.array([[0.2, 0.8], [0.7, 0.0]], dtype=np.float32)

    label = compute_patch_eval_label("patch-a", prediction, truth, threshold=0.5)

    assert label == {
        "pair_id": "patch-a",
        "gt_change_density": pytest.approx(0.5),
        "pred_change_density": pytest.approx(0.5),
        "fn_rate": pytest.approx(0.5),
        "empty_mask_failure": 0,
        "low_response_failure": 0,
    }


def test_compute_patch_eval_label_flags_empty_and_low_response_failures() -> None:
    truth = np.array([[True, True], [False, False]])
    prediction = np.zeros((2, 2), dtype=np.float32)

    label = compute_patch_eval_label("patch-b", prediction, truth, threshold=0.5)

    assert label["gt_change_density"] == pytest.approx(0.5)
    assert label["pred_change_density"] == pytest.approx(0.0)
    assert label["fn_rate"] == pytest.approx(1.0)
    assert label["empty_mask_failure"] == 1
    assert label["low_response_failure"] == 1
