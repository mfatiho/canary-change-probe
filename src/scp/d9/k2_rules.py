"""Frozen falsification-rule application for the D9 K2 no-new-dataset panel.

K2 reuses two already-frozen rules rather than defining a new one: the
manuscript's primary rule (Eq. eq:frozen-rule) for LEVIR-lineage cells, and
the prospective-v2 record for WHU-lineage cells. Neither threshold is tuned
here; this module only evaluates the already-frozen constants against a
label-free canary/evidence ratio pair.
"""

from __future__ import annotations

from typing import Final

PRIMARY_EQ4_RULE = "primary_eq4"
PROSPECTIVE_V2_RULE = "prospective_v2"
FALSIFIED = "F"
NOT_FALSIFIED = "NF"
INCONCLUSIVE = "I"

RULE_THRESHOLDS: Final[dict[str, dict[str, float]]] = {
    PRIMARY_EQ4_RULE: {"c_reject": 0.25, "e_reject": 0.5, "c_nf": 0.6, "e_nf": 0.8},
    PROSPECTIVE_V2_RULE: {"c_reject": 0.10, "e_reject": 0.35, "c_nf": 0.75, "e_nf": 1.00},
}


class K2RuleError(ValueError):
    """Raised when an unknown rule name is requested."""


def apply_rule(canary_ratio: float, evidence_ratio: float, rule_name: str) -> str:
    """Apply a frozen falsification rule to one cell's canary/evidence ratio.

    Args:
        canary_ratio: Source-normalized canary recall ratio (c).
        evidence_ratio: Source-normalized evidence-coverage ratio (e).
        rule_name: One of PRIMARY_EQ4_RULE or PROSPECTIVE_V2_RULE.

    Returns:
        One of FALSIFIED, NOT_FALSIFIED, or INCONCLUSIVE.

    Raises:
        K2RuleError: If rule_name is not a recognized frozen rule.
    """
    try:
        thresholds = RULE_THRESHOLDS[rule_name]
    except KeyError as error:
        raise K2RuleError(f"unknown K2 rule: {rule_name!r}") from error
    if canary_ratio < thresholds["c_reject"] or evidence_ratio < thresholds["e_reject"]:
        return FALSIFIED
    if canary_ratio >= thresholds["c_nf"] and evidence_ratio >= thresholds["e_nf"]:
        return NOT_FALSIFIED
    return INCONCLUSIVE


__all__ = [
    "FALSIFIED",
    "INCONCLUSIVE",
    "NOT_FALSIFIED",
    "PRIMARY_EQ4_RULE",
    "PROSPECTIVE_V2_RULE",
    "RULE_THRESHOLDS",
    "K2RuleError",
    "apply_rule",
]
