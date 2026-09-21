import pytest

from scp.scoring import (
    CombinedScorer,
    DecisionPolicy,
    DecisionThresholds,
    MismatchScorer,
    PassiveScorer,
    ZScoreNormalizer,
)


def test_z_score_normalizer_calibrates_source_statistics() -> None:
    normalizer = ZScoreNormalizer.fit({"density": [1.0, 2.0, 3.0]})

    assert normalizer.transform({"density": 2.0})["density"] == pytest.approx(0.0)


def test_z_score_normalizer_rejects_missing_metric() -> None:
    normalizer = ZScoreNormalizer.fit({"density": [1.0, 2.0]})

    with pytest.raises(KeyError, match="density"):
        normalizer.transform({})


def test_passive_scorer_applies_weights() -> None:
    score = PassiveScorer(weights={"a": 0.25, "b": 0.75}).score({"a": 2.0, "b": 4.0})

    assert score == pytest.approx(3.5)


def test_mismatch_scorer_compares_image_evidence_to_model_response() -> None:
    score = MismatchScorer(image_evidence_key="image", response_key="response").score(
        {"image": 2.0, "response": 0.5}
    )

    assert score == pytest.approx(1.5)


def test_combined_scorer_mixes_passive_and_mismatch_scores() -> None:
    score = CombinedScorer(passive_weight=0.4, mismatch_weight=0.6).score(1.0, 3.0)

    assert score == pytest.approx(2.2)


def test_decision_policy_uses_injected_thresholds() -> None:
    policy = DecisionPolicy(DecisionThresholds(deploy=0.2, reject=0.8))

    assert policy.decide(0.1) == "deploy"
    assert policy.decide(0.5) == "warn"
    assert policy.decide(0.9) == "reject"


def test_decision_thresholds_validate_order() -> None:
    with pytest.raises(ValueError, match="deploy"):
        DecisionThresholds(deploy=0.9, reject=0.1)
