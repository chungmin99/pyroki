"""Diagnostics helpers for headless J-PARSE controller tuning."""

from __future__ import annotations

from dataclasses import dataclass

import jax

try:
    from ._headless_validation import run_jparse_headless_scenarios
except ImportError:  # pragma: no cover - script execution fallback
    from pyroki_snippets._headless_validation import run_jparse_headless_scenarios


@dataclass(frozen=True)
class SweepResult:
    position_gain: float
    orientation_gain: float
    nullspace_gain: float
    all_passed: bool
    medium_final_ori_err: float
    medium_settled: bool


def run_gain_sweep(
    *,
    strictness: str = "default",
    position_gains: tuple[float, ...] = (3.0, 5.0, 7.0),
    orientation_gains: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    nullspace_gains: tuple[float, ...] = (0.0, 0.2, 0.5),
    prefer_cpu: bool = True,
) -> list[SweepResult]:
    """Run a deterministic gain sweep and return ranked summary entries."""
    if prefer_cpu:
        # Keep diagnostics stable and avoid GPU-specific resource failures.
        jax.config.update("jax_platform_name", "cpu")

    results: list[SweepResult] = []

    for position_gain in position_gains:
        for orientation_gain in orientation_gains:
            for nullspace_gain in nullspace_gains:
                output = run_jparse_headless_scenarios(
                    strictness=strictness,  # type: ignore[arg-type]
                    method="jparse",
                    compare_methods=False,
                    controller_profile="default",
                    controller_overrides={
                        "position_gain": position_gain,
                        "orientation_gain": orientation_gain,
                        "nullspace_gain": nullspace_gain,
                    },
                )
                medium = output["scenarios"]["medium_mixed_offset"]
                results.append(
                    SweepResult(
                        position_gain=position_gain,
                        orientation_gain=orientation_gain,
                        nullspace_gain=nullspace_gain,
                        all_passed=bool(output["all_passed"]),
                        medium_final_ori_err=float(medium["final_ori_err"]),
                        medium_settled=bool(medium["settled"]),
                    )
                )

    return sorted(
        results,
        key=lambda item: (
            not item.all_passed,
            item.medium_final_ori_err,
            not item.medium_settled,
            item.nullspace_gain,
        ),
    )


def format_gain_sweep_table(results: list[SweepResult], top_k: int = 10) -> str:
    """Format sweep results into a compact, human-readable table."""
    lines = [
        "pos_gain ori_gain null_gain all_passed medium_ori medium_settled",
        "-------- -------- -------- ----------- ---------- --------------",
    ]
    for row in results[:top_k]:
        lines.append(
            f"{row.position_gain:8.2f} {row.orientation_gain:8.2f} {row.nullspace_gain:8.2f} "
            f"{str(row.all_passed):11s} {row.medium_final_ori_err:10.4f} {str(row.medium_settled):14s}"
        )
    return "\n".join(lines)


def main() -> int:
    results = run_gain_sweep()
    print(format_gain_sweep_table(results, top_k=12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
