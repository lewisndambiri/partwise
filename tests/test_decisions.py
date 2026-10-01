import pytest

from partwise.decisions import Scenario, project_decisions


SCENARIO = {
    "volume": 10000,
    "prevalence": 0.01,
    "missed_defect_cost": 100.0,
    "false_reject_cost": 5.0,
    "review_cost": 1.0,
    "review_capacity": 100,
    "reviewer_sensitivity": 1.0,
    "reviewer_specificity": 1.0,
    "policy": "review_first",
}


def test_review_capacity_and_overflow_account_for_every_flagged_part() -> None:
    scenario = Scenario.from_dict(SCENARIO)
    result = project_decisions(0.8, 0.02, scenario)
    assert result["expected_defective"] == 100
    assert result["flagged_for_action"] == pytest.approx(278)
    assert result["reviewed"] == 100
    assert result["review_overflow"] == pytest.approx(178)
    assert result["missed_defective"] == pytest.approx(20)
    assert result["good_rejected"] == pytest.approx(198 * 178 / 278)
    assert result["total_cost"] == pytest.approx(
        result["cost_misses"] + result["cost_false_rejects"] + result["cost_review"]
    )


def test_auto_reject_has_no_review_cost() -> None:
    scenario = Scenario.from_dict({**SCENARIO, "policy": "auto_reject"})
    result = project_decisions(0.8, 0.02, scenario)
    assert result["reviewed"] == 0
    assert result["good_rejected"] == pytest.approx(198)
    assert result["cost_review"] == 0


def test_scenario_rejects_invalid_prevalence() -> None:
    with pytest.raises(ValueError, match="prevalence"):
        Scenario.from_dict({**SCENARIO, "prevalence": 1.2})
