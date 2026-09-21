import pytest

from scp.d9.k2_rules import (
    FALSIFIED,
    INCONCLUSIVE,
    NOT_FALSIFIED,
    PRIMARY_EQ4_RULE,
    apply_rule,
)


def test_primary_rule_assigns_all_three_states() -> None:
    assert apply_rule(0.20, 0.90, PRIMARY_EQ4_RULE) == FALSIFIED
    assert apply_rule(0.60, 0.80, PRIMARY_EQ4_RULE) == NOT_FALSIFIED
    assert apply_rule(0.40, 0.70, PRIMARY_EQ4_RULE) == INCONCLUSIVE


def test_primary_rule_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError, match="unknown K2 rule"):
        apply_rule(0.5, 0.5, "missing")
