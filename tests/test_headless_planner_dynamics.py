"""Headless dynamics validation for J-PARSE IK and online planning."""

import sys

sys.path.insert(0, "examples")

from pyroki_snippets._headless_validation import (  # pylint: disable=wrong-import-position
    run_jparse_headless_scenarios,
    run_online_planner_headless_scenarios,
)


def _failure_details(result: dict) -> str:
    lines = [
        f"controller={result['controller']} strictness={result['strictness']}",
    ]
    if "thresholds_by_difficulty" in result:
        lines.append(f"thresholds_by_difficulty={result['thresholds_by_difficulty']}")
    for name, metrics in result["scenarios"].items():
        lines.append(f"scenario={name} metrics={metrics}")
    return "\n".join(lines)


def test_jparse_headless_dynamics_stability():
    """Tiered J-PARSE validation: easy/medium strict; hard stability+bounds."""
    result = run_jparse_headless_scenarios(strictness="default")

    scenarios = result["scenarios"]
    easy = scenarios["easy_small_offset"]
    medium = scenarios["medium_mixed_offset"]
    hard = scenarios["hard_large_rotation"]

    # Easy and medium must satisfy strict orientation settling thresholds.
    assert easy["final_ori_err"] <= easy["thresholds"]["final_ori_err"], (
        _failure_details(result)
    )
    assert medium["final_ori_err"] <= medium["thresholds"]["final_ori_err"], (
        _failure_details(result)
    )

    # Hard requires stable behavior with bounded residual, not aggressive accuracy.
    assert hard["sign_changes"] <= hard["thresholds"]["sign_changes_max"], (
        _failure_details(result)
    )
    assert hard["tail_ripple"] <= hard["thresholds"]["tail_ripple_max"], (
        _failure_details(result)
    )
    assert hard["final_ori_err"] <= hard["thresholds"]["final_ori_err"], (
        _failure_details(result)
    )

    # Keep strict overall gate active until all tiered criteria pass.
    assert result["all_passed"], _failure_details(result)


def test_online_planner_headless_dynamics_stability():
    """Online planner replanning loop has acceptable tiered settling behavior."""
    result = run_online_planner_headless_scenarios(strictness="default")

    assert result["all_passed"], _failure_details(result)
    for scenario_name, scenario_metrics in result["scenarios"].items():
        assert scenario_metrics["passed"], (
            f"scenario={scenario_name}\n{_failure_details(result)}"
        )
