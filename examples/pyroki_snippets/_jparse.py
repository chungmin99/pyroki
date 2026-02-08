"""
JAX implementation of the J-PARSE algorithm and velocity IK controller.

J-PARSE (Jacobian-based Projection Algorithm for Resolving Singularities
Effectively) provides singularity-aware inverse kinematics by computing a
modified pseudo-inverse that handles singular configurations smoothly.

Reference: https://github.com/chungmin99/jparse
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
from jax.typing import ArrayLike


def compute_jacobian(
    robot: pk.Robot,
    cfg: ArrayLike,
    target_link_index: int,
    position_only: bool = True,
) -> jnp.ndarray:
    """Compute geometric Jacobian via autodiff on pyroki FK.

    Args:
        robot: PyRoKi robot model.
        cfg: Joint configuration (actuated_count,).
        target_link_index: Index of the target link.
        position_only: If True, return 3xn position Jacobian.
            If False, return 6xn Jacobian (via SE3 log).

    Returns:
        Jacobian matrix. Shape (3, n) if position_only else (6, n).
    """
    cfg = jnp.asarray(cfg)

    if position_only:
        jacobian = jax.jacfwd(
            lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
        )(cfg)[target_link_index]
    else:

        def get_pose_log(q: ArrayLike) -> jnp.ndarray:
            poses = robot.forward_kinematics(q)
            return jaxlie.SE3(poses[target_link_index]).log()

        jacobian = jax.jacfwd(get_pose_log)(cfg)

    return jacobian


def jparse_pseudoinverse(
    jacobian: ArrayLike,
    gamma: float = 0.1,
    singular_direction_gain: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute J-PARSE pseudo-inverse of a Jacobian matrix.

    The J-PARSE algorithm decomposes the Jacobian using SVD and constructs
    a modified pseudo-inverse that:
    1. Clamps singular values below gamma*sigma_max (safety Jacobian)
    2. Projects commands onto non-singular directions (projection Jacobian)
    3. Provides smooth feedback in singular directions

    Formula: J_parse = J_safety^+ @ J_proj @ J_proj^+ + J_safety^+ @ Phi

    Args:
        jacobian: The m x n Jacobian matrix.
        gamma: Singularity threshold in (0, 1). Directions with
            sigma/sigma_max < gamma are treated as singular.
        singular_direction_gain: Gain applied to singular directions.

    Returns:
        Tuple of (J_parse, nullspace_projector):
        - J_parse: n x m J-PARSE pseudo-inverse matrix.
        - nullspace_projector: n x n nullspace projection matrix.
    """
    J = jnp.asarray(jacobian)
    m, n = J.shape

    U, S, Vt = jnp.linalg.svd(J, full_matrices=True)
    k = S.shape[0]  # min(m, n)

    sigma_max = jnp.max(S)
    threshold = gamma * sigma_max

    # Mask: True for non-singular directions (sigma > threshold).
    non_singular = S > threshold

    # Safety singular values: clamp below threshold.
    S_safety = jnp.where(non_singular, S, threshold)

    # Projection singular values: keep only non-singular directions.
    S_proj = jnp.where(non_singular, S, 0.0)

    # Reconstruct matrices from SVD components.
    # J_safety = U[:, :k] @ diag(S_safety) @ Vt[:k, :]
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]

    J_safety = U_k * S_safety[None, :] @ Vt_k
    J_proj = U_k * S_proj[None, :] @ Vt_k

    J_safety_pinv = jnp.linalg.pinv(J_safety)
    J_proj_pinv = jnp.linalg.pinv(J_proj)

    # Singular direction feedback: Phi = sum_i phi_i * u_i @ u_i^T @ Kp
    # where phi_i = sigma_i / (sigma_max * gamma) for singular directions.
    phi = jnp.where(non_singular, 0.0, S / (sigma_max * gamma))
    # Phi_singular = U_k @ diag(phi) @ U_k^T @ Kp
    Kp = singular_direction_gain * jnp.eye(m)
    Phi_singular = (U_k * phi[None, :]) @ U_k.T @ Kp

    J_parse = J_safety_pinv @ J_proj @ J_proj_pinv + J_safety_pinv @ Phi_singular

    # Nullspace projector: N = I - J_safety^+ @ J_safety
    nullspace = jnp.eye(n) - J_safety_pinv @ J_safety

    return J_parse, nullspace


def pinv(jacobian: ArrayLike) -> jnp.ndarray:
    """Standard Moore-Penrose pseudo-inverse (for comparison)."""
    return jnp.linalg.pinv(jnp.asarray(jacobian))


def damped_least_squares(
    jacobian: ArrayLike,
    damping: float = 0.05,
) -> jnp.ndarray:
    """Damped least squares (Levenberg-Marquardt) pseudo-inverse.

    Args:
        jacobian: The m x n Jacobian matrix.
        damping: Damping factor lambda.

    Returns:
        The n x m DLS pseudo-inverse.
    """
    J = jnp.asarray(jacobian)
    n = J.shape[1]
    return jnp.linalg.inv(J.T @ J + damping**2 * jnp.eye(n)) @ J.T


def manipulability_measure(jacobian: ArrayLike) -> jnp.ndarray:
    """Yoshikawa's manipulability measure: sqrt(det(J @ J^T))."""
    J = jnp.asarray(jacobian)
    return jnp.sqrt(jnp.linalg.det(J @ J.T))


def inverse_condition_number(jacobian: ArrayLike) -> jnp.ndarray:
    """Inverse condition number: sigma_min / sigma_max."""
    J = jnp.asarray(jacobian)
    S = jnp.linalg.svd(J, compute_uv=False)
    return jnp.min(S) / jnp.max(S)


def jparse_step(
    robot: pk.Robot,
    cfg: ArrayLike,
    target_link_index: int,
    target_position: ArrayLike,
    *,
    method: Literal["jparse", "pinv", "dls"] = "jparse",
    gamma: float = 0.1,
    singular_direction_gain: float = 1.0,
    position_gain: float = 5.0,
    nullspace_gain: float = 0.5,
    max_joint_velocity: float = 2.0,
    dls_damping: float = 0.05,
    dt: float = 0.02,
    home_cfg: ArrayLike | None = None,
) -> tuple[np.ndarray, dict]:
    """Single velocity IK step using J-PARSE (or pinv/DLS).

    Computes Jacobian -> pseudo-inverse -> joint velocities -> integrate -> clamp.

    Args:
        robot: PyRoKi robot model.
        cfg: Current joint configuration (actuated_count,).
        target_link_index: Index of the target link.
        target_position: Target position (3,).
        method: IK method — "jparse", "pinv", or "dls".
        gamma: J-PARSE singularity threshold.
        singular_direction_gain: Gain for singular directions.
        position_gain: Proportional gain for position error.
        nullspace_gain: Gain for nullspace motion toward home.
        max_joint_velocity: Maximum joint velocity (rad/s).
        dls_damping: Damping factor for DLS method.
        dt: Time step for integration.
        home_cfg: Home configuration for nullspace. If None, uses joint midpoints.

    Returns:
        Tuple of (new_cfg, info):
        - new_cfg: New joint configuration after integration (numpy).
        - info: Dict with position_error, max_joint_vel, jacobian,
            manipulability, inverse_condition_number.
    """
    cfg = jnp.asarray(cfg)
    target_position = jnp.asarray(target_position)

    # Current end-effector position.
    poses = robot.forward_kinematics(cfg)
    current_pos = jaxlie.SE3(poses[target_link_index]).translation()

    # Position error and desired task-space velocity.
    pos_error = target_position - current_pos
    pos_error_mag = float(jnp.linalg.norm(pos_error))
    v_des = position_gain * pos_error

    # Compute Jacobian.
    jacobian = compute_jacobian(robot, cfg, target_link_index, position_only=True)

    # Compute pseudo-inverse and nullspace.
    if method == "jparse":
        J_inv, N = jparse_pseudoinverse(jacobian, gamma, singular_direction_gain)
    elif method == "pinv":
        J_inv = pinv(jacobian)
        N = jnp.eye(jacobian.shape[1]) - J_inv @ jacobian
    else:  # dls
        J_inv = damped_least_squares(jacobian, dls_damping)
        N = jnp.eye(jacobian.shape[1]) - J_inv @ jacobian

    # Primary task joint velocities.
    dq = J_inv @ v_des

    # Nullspace motion toward home configuration.
    if nullspace_gain > 0:
        if home_cfg is None:
            lower = robot.joints.lower_limits
            upper = robot.joints.upper_limits
            home = (lower + upper) / 2.0
        else:
            home = jnp.asarray(home_cfg)
        dq_null = N @ (-nullspace_gain * (cfg - home))
        dq = dq + dq_null

    # Track raw max velocity before limiting.
    max_joint_vel = float(jnp.max(jnp.abs(dq)))

    # Apply velocity limits.
    scale = jnp.where(
        jnp.max(jnp.abs(dq)) > max_joint_velocity,
        max_joint_velocity / jnp.max(jnp.abs(dq)),
        1.0,
    )
    dq = dq * scale

    # Integrate.
    new_cfg = cfg + dq * dt

    # Clamp to joint limits.
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    new_cfg = jnp.clip(new_cfg, lower, upper)

    info = {
        "position_error": pos_error_mag,
        "max_joint_vel": max_joint_vel,
        "jacobian": np.asarray(jacobian),
        "manipulability": float(manipulability_measure(jacobian)),
        "inverse_condition_number": float(inverse_condition_number(jacobian)),
    }

    return np.asarray(new_cfg), info
