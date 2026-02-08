"""Headless stability validation for J-PARSE IK and online planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger
from pyroki_snippets._headless_validation import (
    available_jparse_profiles,
    run_jparse_headless_scenarios,
    run_online_planner_headless_scenarios,
)


def _print_method_summary(title: str, result: dict[str, Any]) -> None:
    print(
        f"\n[{title}] method={result.get('method', 'n/a')} "
        f"strictness={result['strictness']} all_passed={result['all_passed']}"
    )

    for scenario_name, scenario in result["scenarios"].items():
        thresholds = scenario["thresholds"]
        print(
            f"  - {scenario_name:24s} diff={scenario['difficulty']:6s} "
            f"pass={scenario['passed']} "
            f"final_pos={scenario['final_pos_err']:.4f}/{thresholds['final_pos_err']:.4f} "
            f"final_ori={scenario['final_ori_err']:.4f}/{thresholds['final_ori_err']:.4f} "
            f"settled={scenario['settled']}@{scenario['settling_step']}"
        )


def _print_summary(title: str, result: dict[str, Any]) -> None:
    if result.get("compare_methods", False):
        print(f"\n[{title}] compare_methods=True all_passed={result['all_passed']}")
        for method_name, method_result in result["methods"].items():
            _print_method_summary(f"{title}:{method_name}", method_result)
        return

    _print_method_summary(title, result)


def _dump_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote JSON report to: {output_path}")


def main() -> int:
    logger.disable("jaxls")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["jparse", "online", "both"],
        default="both",
        help="Which controller(s) to validate.",
    )
    parser.add_argument(
        "--strictness",
        choices=["default", "relaxed"],
        default="default",
        help="Threshold strictness profile.",
    )
    parser.add_argument(
        "--jparse-steps",
        type=int,
        default=180,
        help="Number of closed-loop steps for J-PARSE scenarios.",
    )
    parser.add_argument(
        "--replans",
        type=int,
        default=20,
        help="Number of receding-horizon replans for online scenarios.",
    )
    parser.add_argument(
        "--method",
        choices=["jparse", "pinv", "dls"],
        default="jparse",
        help="Velocity IK method when not in compare mode.",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Run and report jparse/pinv/dls side-by-side.",
    )
    parser.add_argument(
        "--controller-profile",
        choices=list(available_jparse_profiles()),
        default="default",
        help="Controller gain profile for velocity IK.",
    )
    parser.add_argument(
        "--dump-json",
        default=None,
        help="Optional path to dump full results JSON.",
    )
    args = parser.parse_args()

    aggregated_results: dict[str, Any] = {}
    overall_flags: list[bool] = []

    if args.only in {"jparse", "both"}:
        jparse_result = run_jparse_headless_scenarios(
            strictness=args.strictness,
            steps=args.jparse_steps,
            method=args.method,
            compare_methods=args.compare_methods,
            controller_profile=args.controller_profile,
        )
        _print_summary("J-PARSE", jparse_result)
        aggregated_results["jparse"] = jparse_result
        overall_flags.append(bool(jparse_result["all_passed"]))

    if args.only in {"online", "both"}:
        online_result = run_online_planner_headless_scenarios(
            strictness=args.strictness,
            replans=args.replans,
        )
        _print_summary("ONLINE", online_result)
        aggregated_results["online"] = online_result
        overall_flags.append(bool(online_result["all_passed"]))

    all_passed = all(overall_flags)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")

    if args.dump_json:
        payload = {
            "strictness": args.strictness,
            "controller_profile": args.controller_profile,
            "compare_methods": args.compare_methods,
            "results": aggregated_results,
            "all_passed": all_passed,
        }
        _dump_json(args.dump_json, payload)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
