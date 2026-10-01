"""Translate measured inspection errors into an explicit production scenario."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    volume: int
    prevalence: float
    missed_defect_cost: float
    false_reject_cost: float
    review_cost: float
    review_capacity: int
    reviewer_sensitivity: float
    reviewer_specificity: float
    policy: str = "review_first"

    @classmethod
    def from_dict(cls, values: dict) -> "Scenario":
        scenario = cls(**values)
        if scenario.volume < 1 or scenario.review_capacity < 0:
            raise ValueError("volume must be positive and review_capacity nonnegative")
        for name in ("prevalence", "reviewer_sensitivity", "reviewer_specificity"):
            value = getattr(scenario, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("missed_defect_cost", "false_reject_cost", "review_cost"):
            if getattr(scenario, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if scenario.policy not in {"review_first", "auto_reject"}:
            raise ValueError("policy must be review_first or auto_reject")
        return scenario


def project_decisions(
    defect_recall: float,
    false_reject_rate: float,
    scenario: Scenario,
) -> dict:
    """Project expected counts; flagged items are sampled proportionally for review.

    A reviewed flagged item may be accepted or rejected by the human. Overflow
    beyond review capacity is automatically rejected. Unflagged items pass.
    This is a scenario calculation, not an estimate of actual factory costs.
    """
    for name, rate in (("defect_recall", defect_recall), ("false_reject_rate", false_reject_rate)):
        if not 0 <= rate <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    defective = scenario.volume * scenario.prevalence
    good = scenario.volume - defective
    flagged_defective = defective * defect_recall
    flagged_good = good * false_reject_rate
    flagged = flagged_defective + flagged_good
    initially_missed = defective - flagged_defective

    reviewed = min(flagged, scenario.review_capacity) if scenario.policy == "review_first" else 0.0
    review_fraction = reviewed / flagged if flagged else 0.0
    reviewed_defective = flagged_defective * review_fraction
    reviewed_good = flagged_good * review_fraction
    overflow_defective = flagged_defective - reviewed_defective
    overflow_good = flagged_good - reviewed_good

    missed_defective = initially_missed + reviewed_defective * (1 - scenario.reviewer_sensitivity)
    good_rejected = overflow_good + reviewed_good * (1 - scenario.reviewer_specificity)
    correctly_rejected_defective = overflow_defective + reviewed_defective * scenario.reviewer_sensitivity
    cost_misses = missed_defective * scenario.missed_defect_cost
    cost_false_rejects = good_rejected * scenario.false_reject_cost
    cost_review = reviewed * scenario.review_cost

    return {
        "expected_parts": scenario.volume,
        "expected_defective": defective,
        "expected_good": good,
        "flagged_for_action": flagged,
        "reviewed": reviewed,
        "review_overflow": flagged - reviewed,
        "missed_defective": missed_defective,
        "good_rejected": good_rejected,
        "correctly_rejected_defective": correctly_rejected_defective,
        "cost_misses": cost_misses,
        "cost_false_rejects": cost_false_rejects,
        "cost_review": cost_review,
        "total_cost": cost_misses + cost_false_rejects + cost_review,
        "cost_per_part": (cost_misses + cost_false_rejects + cost_review) / scenario.volume,
    }
