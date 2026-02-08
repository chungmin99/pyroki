from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
from pyroki.collision import HalfSpace, RobotCollision, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description

from ._jparse import jparse_step
from ._online_planning import solve_online_planning

EPS = 1e-9


@dataclass(frozen=True)
class StabilityThresholds:
    final_pos_err: float
    final_ori_err: float
    settling_step_max: int
    sign_changes_max: int
    tail_ripple_max: float


@dataclass(frozen=True)
class SettlingConfig:
    hold_steps: int
    tail_window: int
    derivative_deadband: float
    warmup_steps: int


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    dpos: np.ndarray
    axis: np.ndarray
    angle: float


@dataclass(frozen=True)
class ControllerProfile:
    gamma: float
    singular_direction_gain_position: float
    singular_direction_gain_angular: float
    position_gain: float
    orientation_gain: float
    nullspace_gain: float
    max_joint_velocity: float
    dls_damping: float


_JPARSE_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        name="easy_small_offset",
        difficulty="easy",
        dpos=np.array([0.03, 0.00, 0.00]),
        axis=np.array([0.0, 0.0, 1.0]),
        angle=0.15,
    ),
    ScenarioSpec(
        name="medium_mixed_offset",
        difficulty="medium",
        dpos=np.array([0.06, -0.04, 0.03]),
        axis=np.array([0.0, 1.0, 0.0]),
        angle=0.35,
    ),
    ScenarioSpec(
        name="hard_large_rotation",
        difficulty="hard",
        dpos=np.array([0.08, 0.04, -0.02]),
        axis=np.array([1.0, 0.0, 0.0]),
        angle=0.60,
    ),
)

_ONLINE_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        name="easy_static_obstacle",
        difficulty="easy",
        dpos=np.array([-0.09, 0.02, -0.04]),
        axis=np.array([0.0, 0.0, 1.0]),
        angle=0.20,
    ),
    ScenarioSpec(
        name="medium_near_path_obstacle",
        difficulty="medium",
        dpos=np.array([-0.13, -0.06, -0.06]),
        axis=np.array([0.0, 1.0, 0.0]),
        angle=0.35,
    ),
    ScenarioSpec(
        name="hard_crossing_obstacle",
        difficulty="hard",
        dpos=np.array([-0.15, 0.07, -0.08]),
        axis=np.array([1.0, 0.0, 0.0]),
        angle=0.50,
    ),
)

_JPARSE_PROFILES: dict[str, ControllerProfile] = {
    "default": ControllerProfile(
        gamma=0.1,
        singular_direction_gain_position=2.0,
        singular_direction_gain_angular=2.0,
        position_gain=5.0,
        orientation_gain=1.0,
        nullspace_gain=0.5,
        max_joint_velocity=2.0,
        dls_damping=0.05,
    ),
    "aggressive_orientation": ControllerProfile(
        gamma=0.1,
        singular_direction_gain_position=2.0,
        singular_direction_gain_angular=3.0,
        position_gain=3.5,
        orientation_gain=2.0,
        nullspace_gain=0.2,
        max_joint_velocity=2.2,
        dls_damping=0.05,
    ),
}


def available_jparse_profiles() -> tuple[str, ...]:
    return tuple(_JPARSE_PROFILES.keys())


def _normalize_quat(wxyz: np.ndarray) -> np.ndarray:
    wxyz = np.asarray(wxyz, dtype=np.float64)
    return wxyz / max(np.linalg.norm(wxyz), EPS)


def _build_target_wxyz(
    base_wxyz: np.ndarray, axis_xyz: np.ndarray, angle_rad: float
) -> np.ndarray:
    axis = np.asarray(axis_xyz, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), EPS)
    delta = jaxlie.SO3.exp(jnp.asarray(axis * angle_rad))
    base = jaxlie.SO3(jnp.asarray(_normalize_quat(base_wxyz)))
    return np.asarray((delta @ base).wxyz)


def _orientation_error_norm(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> float:
    q_current = jaxlie.SO3(jnp.asarray(_normalize_quat(current_wxyz)))
    q_target = jaxlie.SO3(jnp.asarray(_normalize_quat(target_wxyz)))
    return float(jnp.linalg.norm((q_target @ q_current.inverse()).log()))


def _pose_errors(
    robot: pk.Robot,
    cfg: np.ndarray,
    target_link_index: int,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
) -> tuple[float, float]:
    pose = jaxlie.SE3(robot.forward_kinematics(jnp.asarray(cfg))[target_link_index])
    pos_err = float(
        np.linalg.norm(np.asarray(target_position) - np.asarray(pose.translation()))
    )
    ori_err = _orientation_error_norm(np.asarray(pose.rotation().wxyz), target_wxyz)
    return pos_err, ori_err


def compute_settling_metrics(
    combined_error: np.ndarray,
    position_error: np.ndarray,
    orientation_error: np.ndarray,
    *,
    position_tolerance: float,
    orientation_tolerance: float,
    hold_steps: int,
    tail_window: int,
    derivative_deadband: float,
    warmup_steps: int,
) -> dict[str, float | int | bool]:
    combined = np.asarray(combined_error, dtype=np.float64)
    pos = np.asarray(position_error, dtype=np.float64)
    ori = np.asarray(orientation_error, dtype=np.float64)

    if combined.ndim != 1 or pos.ndim != 1 or ori.ndim != 1:
        raise ValueError("Error series must be 1D arrays.")
    if not (len(combined) == len(pos) == len(ori)):
        raise ValueError("Error series must have the same length.")
    if len(combined) == 0:
        raise ValueError("Error series cannot be empty.")

    if len(combined) == 1:
        sign_changes = 0
    else:
        diffs = np.diff(combined[max(0, warmup_steps) :])
        diffs[np.abs(diffs) < derivative_deadband] = 0.0
        signs = np.sign(diffs)
        signs = signs[signs != 0.0]
        sign_changes = (
            int(np.sum(signs[1:] * signs[:-1] < 0.0)) if len(signs) >= 2 else 0
        )

    tail = combined[max(0, len(combined) - tail_window) :]
    tail_ripple = float(
        (np.max(tail) - np.min(tail)) / max(float(np.max(combined)), EPS)
    )

    settling_step = -1
    if hold_steps > 0 and len(combined) >= hold_steps:
        for start in range(0, len(combined) - hold_steps + 1):
            end = start + hold_steps
            if np.all(pos[start:end] <= position_tolerance) and np.all(
                ori[start:end] <= orientation_tolerance
            ):
                settling_step = start
                break

    return {
        "final_pos_err": float(pos[-1]),
        "final_ori_err": float(ori[-1]),
        "peak_combined_err": float(np.max(combined)),
        "sign_changes": sign_changes,
        "tail_ripple": tail_ripple,
        "settled": settling_step >= 0,
        "settling_step": settling_step,
    }


def _thresholds_with_strictness(
    thresholds: StabilityThresholds,
    strictness: Literal["default", "relaxed"],
) -> StabilityThresholds:
    if strictness == "default":
        return thresholds
    if strictness != "relaxed":
        raise ValueError(f"Unsupported strictness: {strictness}")

    return StabilityThresholds(
        final_pos_err=thresholds.final_pos_err * 1.5,
        final_ori_err=thresholds.final_ori_err * 1.5,
        settling_step_max=int(round(thresholds.settling_step_max * 1.25)),
        sign_changes_max=int(round(thresholds.sign_changes_max * 1.5)),
        tail_ripple_max=thresholds.tail_ripple_max * 1.5,
    )


def _serialize_threshold_map(
    threshold_map: dict[str, StabilityThresholds],
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "final_pos_err": value.final_pos_err,
            "final_ori_err": value.final_ori_err,
            "settling_step_max": value.settling_step_max,
            "sign_changes_max": value.sign_changes_max,
            "tail_ripple_max": value.tail_ripple_max,
        }
        for key, value in threshold_map.items()
    }


def _controller_profile(
    profile_name: str,
    overrides: dict[str, float] | None,
) -> ControllerProfile:
    if profile_name not in _JPARSE_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile_name}'. Available: {sorted(_JPARSE_PROFILES)}"
        )

    profile_data = asdict(_JPARSE_PROFILES[profile_name])
    if overrides:
        unknown_keys = set(overrides.keys()) - set(profile_data.keys())
        if unknown_keys:
            raise ValueError(f"Unknown override keys: {sorted(unknown_keys)}")
        profile_data.update(overrides)

    return ControllerProfile(**profile_data)


def _add_failure_classifiers(
    metrics: dict[str, float | int | bool],
    pos_errors: np.ndarray,
    ori_errors: np.ndarray,
    *,
    thresholds: StabilityThresholds,
    settling_cfg: SettlingConfig,
) -> None:
    tail_window = max(3, min(settling_cfg.tail_window, len(ori_errors)))
    ori_tail = ori_errors[-tail_window:]
    pos_tail = pos_errors[-tail_window:]

    ori_diffs = np.diff(ori_tail) if len(ori_tail) >= 2 else np.array([0.0])
    plateau = bool(
        np.all(np.abs(ori_diffs) <= settling_cfg.derivative_deadband)
        and float(metrics["final_ori_err"]) > thresholds.final_ori_err
    )

    competing_objectives = bool(
        np.mean(pos_tail <= thresholds.final_pos_err) >= 0.7
        and float(metrics["final_ori_err"]) > thresholds.final_ori_err
        and np.mean(ori_diffs) > -settling_cfg.derivative_deadband
    )

    metrics["plateau"] = plateau
    metrics["competing_objectives"] = competing_objectives


def _evaluate_thresholds(
    metrics: dict[str, float | int | bool],
    thresholds: StabilityThresholds,
) -> bool:
    return bool(
        metrics["final_pos_err"] <= thresholds.final_pos_err
        and metrics["final_ori_err"] <= thresholds.final_ori_err
        and metrics["settled"]
        and metrics["settling_step"] <= thresholds.settling_step_max
        and metrics["sign_changes"] <= thresholds.sign_changes_max
        and metrics["tail_ripple"] <= thresholds.tail_ripple_max
    )


def _run_jparse_method(
    *,
    method: Literal["jparse", "pinv", "dls"],
    strictness: Literal["default", "relaxed"],
    steps: int,
    dt: float,
    capture_trace: bool,
    profile: ControllerProfile,
) -> dict[str, Any]:
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)
    target_link_index = robot.links.names.index("panda_hand")

    cfg0 = np.asarray((robot.joints.lower_limits + robot.joints.upper_limits) / 2.0)
    init_pose = jaxlie.SE3(
        robot.forward_kinematics(jnp.asarray(cfg0))[target_link_index]
    )
    init_pos = np.asarray(init_pose.translation())
    init_wxyz = np.asarray(init_pose.rotation().wxyz)

    threshold_by_difficulty = {
        "easy": StabilityThresholds(
            final_pos_err=0.02,
            final_ori_err=0.16,
            settling_step_max=140,
            sign_changes_max=14,
            tail_ripple_max=0.18,
        ),
        "medium": StabilityThresholds(
            final_pos_err=0.02,
            final_ori_err=0.20,
            settling_step_max=150,
            sign_changes_max=14,
            tail_ripple_max=0.18,
        ),
        "hard": StabilityThresholds(
            final_pos_err=0.03,
            final_ori_err=0.35,
            settling_step_max=170,
            sign_changes_max=16,
            tail_ripple_max=0.22,
        ),
    }
    threshold_by_difficulty = {
        key: _thresholds_with_strictness(value, strictness)
        for key, value in threshold_by_difficulty.items()
    }

    settling_cfg = SettlingConfig(
        hold_steps=10,
        tail_window=20,
        derivative_deadband=1e-4,
        warmup_steps=10,
    )

    scenario_results: dict[str, dict[str, Any]] = {}
    for scenario in _JPARSE_SCENARIOS:
        cfg = cfg0.copy()
        target_pos = init_pos + scenario.dpos
        target_wxyz = _build_target_wxyz(init_wxyz, scenario.axis, scenario.angle)

        pos_errors: list[float] = []
        ori_errors: list[float] = []
        combined_errors: list[float] = []
        dq_inf_series: list[float] = []
        saturation_series: list[float] = []
        inverse_condition_series: list[float] = []

        thresholds = threshold_by_difficulty[scenario.difficulty]
        for _ in range(steps):
            cfg, info = jparse_step(
                robot=robot,
                cfg=cfg,
                target_link_index=target_link_index,
                target_position=target_pos,
                target_wxyz=target_wxyz,
                method=method,
                gamma=profile.gamma,
                singular_direction_gain_position=profile.singular_direction_gain_position,
                singular_direction_gain_angular=profile.singular_direction_gain_angular,
                position_gain=profile.position_gain,
                orientation_gain=profile.orientation_gain,
                nullspace_gain=profile.nullspace_gain,
                max_joint_velocity=profile.max_joint_velocity,
                dls_damping=profile.dls_damping,
                dt=dt,
            )

            pos_err, ori_err = _pose_errors(
                robot,
                cfg,
                target_link_index,
                target_pos,
                target_wxyz,
            )
            pos_errors.append(pos_err)
            ori_errors.append(ori_err)

            combined_errors.append(
                float(
                    np.sqrt(
                        (pos_err / thresholds.final_pos_err) ** 2
                        + (ori_err / thresholds.final_ori_err) ** 2
                    )
                )
            )

            dq_inf = float(info["max_joint_vel"])
            dq_inf_series.append(dq_inf)
            saturation_series.append(
                min(dq_inf / max(profile.max_joint_velocity, EPS), 1.0)
            )
            inverse_condition_series.append(float(info["inverse_condition_number"]))

        pos_arr = np.asarray(pos_errors)
        ori_arr = np.asarray(ori_errors)
        combined_arr = np.asarray(combined_errors)

        metrics = compute_settling_metrics(
            combined_arr,
            pos_arr,
            ori_arr,
            position_tolerance=thresholds.final_pos_err,
            orientation_tolerance=thresholds.final_ori_err,
            hold_steps=settling_cfg.hold_steps,
            tail_window=settling_cfg.tail_window,
            derivative_deadband=settling_cfg.derivative_deadband,
            warmup_steps=settling_cfg.warmup_steps,
        )
        _add_failure_classifiers(
            metrics,
            pos_arr,
            ori_arr,
            thresholds=thresholds,
            settling_cfg=settling_cfg,
        )
        metrics["mean_saturation"] = float(np.mean(saturation_series))
        metrics["peak_saturation"] = float(np.max(saturation_series))
        metrics["mean_inverse_condition"] = float(np.mean(inverse_condition_series))
        metrics["passed"] = _evaluate_thresholds(metrics, thresholds)

        scenario_result: dict[str, Any] = {
            "difficulty": scenario.difficulty,
            "thresholds": {
                "final_pos_err": thresholds.final_pos_err,
                "final_ori_err": thresholds.final_ori_err,
                "settling_step_max": thresholds.settling_step_max,
                "sign_changes_max": thresholds.sign_changes_max,
                "tail_ripple_max": thresholds.tail_ripple_max,
            },
            **metrics,
        }

        if capture_trace:
            scenario_result["trace"] = {
                "position_error": pos_errors,
                "orientation_error": ori_errors,
                "combined_error": combined_errors,
                "dq_inf": dq_inf_series,
                "saturation": saturation_series,
                "inverse_condition_number": inverse_condition_series,
            }

        scenario_results[scenario.name] = scenario_result

    return {
        "controller": "jparse",
        "method": method,
        "strictness": strictness,
        "profile": asdict(profile),
        "thresholds_by_difficulty": _serialize_threshold_map(threshold_by_difficulty),
        "settling_config": asdict(settling_cfg),
        "scenarios": scenario_results,
        "all_passed": all(result["passed"] for result in scenario_results.values()),
    }


def run_jparse_headless_scenarios(
    *,
    strictness: Literal["default", "relaxed"] = "default",
    steps: int = 180,
    dt: float = 0.02,
    method: Literal["jparse", "pinv", "dls"] = "jparse",
    compare_methods: bool = False,
    controller_profile: str = "default",
    controller_overrides: dict[str, float] | None = None,
    capture_trace: bool = False,
) -> dict[str, Any]:
    profile = _controller_profile(controller_profile, controller_overrides)

    if not compare_methods:
        return _run_jparse_method(
            method=method,
            strictness=strictness,
            steps=steps,
            dt=dt,
            capture_trace=capture_trace,
            profile=profile,
        )

    methods: tuple[Literal["jparse", "pinv", "dls"], ...] = ("jparse", "pinv", "dls")
    method_results = {
        method_name: _run_jparse_method(
            method=method_name,
            strictness=strictness,
            steps=steps,
            dt=dt,
            capture_trace=capture_trace,
            profile=profile,
        )
        for method_name in methods
    }

    return {
        "controller": "velocity_ik",
        "compare_methods": True,
        "strictness": strictness,
        "profile": asdict(profile),
        "methods": method_results,
        "all_passed": all(result["all_passed"] for result in method_results.values()),
    }


def _online_obstacle_position(
    scenario_name: str,
    step: int,
    replans: int,
) -> np.ndarray:
    if scenario_name == "easy_static_obstacle":
        return np.array([0.15, 0.45, 0.32])
    if scenario_name == "medium_near_path_obstacle":
        return np.array([0.32, 0.12, 0.36])
    if scenario_name == "hard_crossing_obstacle":
        alpha = 0.0 if replans <= 1 else step / (replans - 1)
        y = 0.22 * (1.0 - 2.0 * alpha)
        return np.array([0.35, y, 0.34])
    raise ValueError(f"Unknown scenario: {scenario_name}")


def run_online_planner_headless_scenarios(
    *,
    strictness: Literal["default", "relaxed"] = "default",
    replans: int = 20,
    timesteps: int = 5,
    dt: float = 0.1,
    capture_trace: bool = False,
) -> dict[str, Any]:
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)
    target_link_name = "panda_hand"
    target_link_index = robot.links.names.index(target_link_name)

    cfg0 = np.asarray((robot.joints.lower_limits + robot.joints.upper_limits) / 2.0)
    init_pose = jaxlie.SE3(
        robot.forward_kinematics(jnp.asarray(cfg0))[target_link_index]
    )
    init_pos = np.asarray(init_pose.translation())
    init_wxyz = np.asarray(init_pose.rotation().wxyz)

    plane = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    base_sphere = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    threshold_by_difficulty = {
        "easy": StabilityThresholds(
            final_pos_err=0.03,
            final_ori_err=0.15,
            settling_step_max=18,
            sign_changes_max=10,
            tail_ripple_max=0.22,
        ),
        "medium": StabilityThresholds(
            final_pos_err=0.03,
            final_ori_err=0.15,
            settling_step_max=18,
            sign_changes_max=10,
            tail_ripple_max=0.22,
        ),
        "hard": StabilityThresholds(
            final_pos_err=0.04,
            final_ori_err=0.20,
            settling_step_max=20,
            sign_changes_max=12,
            tail_ripple_max=0.25,
        ),
    }
    threshold_by_difficulty = {
        key: _thresholds_with_strictness(value, strictness)
        for key, value in threshold_by_difficulty.items()
    }

    settling_cfg = SettlingConfig(
        hold_steps=4,
        tail_window=6,
        derivative_deadband=5e-4,
        warmup_steps=2,
    )

    scenario_results: dict[str, dict[str, Any]] = {}
    for scenario in _ONLINE_SCENARIOS:
        cfg = cfg0.copy()
        prev_sols = np.asarray(
            robot.joint_var_cls.default_factory()[None].repeat(timesteps, axis=0)
        )

        target_pos = init_pos + scenario.dpos
        target_wxyz = _build_target_wxyz(init_wxyz, scenario.axis, scenario.angle)

        pos_errors: list[float] = []
        ori_errors: list[float] = []
        combined_errors: list[float] = []

        thresholds = threshold_by_difficulty[scenario.difficulty]
        for step in range(replans):
            sphere_pos = _online_obstacle_position(scenario.name, step, replans)
            sphere_world = base_sphere.transform_from_wxyz_position(
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=sphere_pos,
            )

            sol_traj, _, _ = solve_online_planning(
                robot=robot,
                robot_coll=robot_coll,
                world_coll=[plane, sphere_world],
                target_link_name=target_link_name,
                target_position=target_pos,
                target_wxyz=target_wxyz,
                timesteps=timesteps,
                dt=dt,
                start_cfg=cfg,
                prev_sols=prev_sols,
            )

            cfg = np.asarray(sol_traj[0])
            prev_sols = np.asarray(sol_traj)

            pos_err, ori_err = _pose_errors(
                robot,
                cfg,
                target_link_index,
                target_pos,
                target_wxyz,
            )
            pos_errors.append(pos_err)
            ori_errors.append(ori_err)
            combined_errors.append(
                float(
                    np.sqrt(
                        (pos_err / thresholds.final_pos_err) ** 2
                        + (ori_err / thresholds.final_ori_err) ** 2
                    )
                )
            )

        pos_arr = np.asarray(pos_errors)
        ori_arr = np.asarray(ori_errors)
        combined_arr = np.asarray(combined_errors)
        metrics = compute_settling_metrics(
            combined_arr,
            pos_arr,
            ori_arr,
            position_tolerance=thresholds.final_pos_err,
            orientation_tolerance=thresholds.final_ori_err,
            hold_steps=settling_cfg.hold_steps,
            tail_window=settling_cfg.tail_window,
            derivative_deadband=settling_cfg.derivative_deadband,
            warmup_steps=settling_cfg.warmup_steps,
        )
        _add_failure_classifiers(
            metrics,
            pos_arr,
            ori_arr,
            thresholds=thresholds,
            settling_cfg=settling_cfg,
        )
        metrics["passed"] = _evaluate_thresholds(metrics, thresholds)

        scenario_result: dict[str, Any] = {
            "difficulty": scenario.difficulty,
            "thresholds": {
                "final_pos_err": thresholds.final_pos_err,
                "final_ori_err": thresholds.final_ori_err,
                "settling_step_max": thresholds.settling_step_max,
                "sign_changes_max": thresholds.sign_changes_max,
                "tail_ripple_max": thresholds.tail_ripple_max,
            },
            **metrics,
        }

        if capture_trace:
            scenario_result["trace"] = {
                "position_error": pos_errors,
                "orientation_error": ori_errors,
                "combined_error": combined_errors,
            }

        scenario_results[scenario.name] = scenario_result

    return {
        "controller": "online_planner",
        "strictness": strictness,
        "thresholds_by_difficulty": _serialize_threshold_map(threshold_by_difficulty),
        "settling_config": asdict(settling_cfg),
        "scenarios": scenario_results,
        "all_passed": all(result["passed"] for result in scenario_results.values()),
    }
