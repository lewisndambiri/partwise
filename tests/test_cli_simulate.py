import json
from pathlib import Path

from partwise.cli import main


def test_simulation_cli_preserves_assumptions_and_sample_sizes(tmp_path: Path) -> None:
    result = tmp_path / "baseline.json"
    scenario = tmp_path / "scenario.json"
    output = tmp_path / "projection.json"
    result.write_text(json.dumps({
        "method": "example",
        "threshold": 0.2,
        "metrics": {
            "defect_recall": 0.8,
            "false_reject_rate": 0.02,
            "n_good": 22,
            "n_defective": 93,
        },
    }))
    scenario.write_text(json.dumps({
        "volume": 10000,
        "prevalence": 0.01,
        "missed_defect_cost": 100.0,
        "false_reject_cost": 5.0,
        "review_cost": 1.0,
        "review_capacity": 100,
        "reviewer_sensitivity": 1.0,
        "reviewer_specificity": 1.0,
        "policy": "review_first",
    }))

    assert main([
        "simulate", "--result", str(result), "--scenario", str(scenario),
        "--output", str(output),
    ]) == 0
    payload = json.loads(output.read_text())
    assert payload["basis"]["test_good_count"] == 22
    assert payload["basis"]["test_defective_count"] == 93
    assert payload["scenario"]["review_capacity"] == 100
    assert payload["projection"]["reviewed"] == 100
