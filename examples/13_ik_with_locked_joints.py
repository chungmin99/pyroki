"""IK with Locked Joints

Demonstrates and benchmarks two joint locking approaches during IK optimization:
1. Masked Jacobian: Zeros out Jacobian columns for locked joints (approach in _solve_ik_masked)
2. Masked Retraction: Uses custom retract_fn to filter delta updates (approach in _solve_ik_retract_masked)

Uses checkboxes to dynamically lock/unlock individual joints without triggering JAX recompilation.
"""

import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


def main():
    """Main function for IK with locked joints."""

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )

    # Create GUI folder for approach selection and metrics.
    approach_dropdown = server.gui.add_dropdown(
        "Joint locking",
        options=["Masked Jacobian", "Masked Retraction"],
        initial_value="Masked Jacobian",
    )

    # Create GUI folder for joint locking controls.
    reset_requested = [False]  # Use list to allow mutation in callback
    with server.gui.add_folder("Optimization"):
        # Add button to lock/unlock all.
        lock_all_btn = server.gui.add_button_group(
            label="Control (all)", options=["Lock", "Unlock", "Reset"]
        )

        # Create checkboxes for each joint (checked = unlocked/optimized).
        joint_checkboxes = []
        with server.gui.add_folder("Robot Joints"):
            for i, name in enumerate(robot.joints.actuated_names):
                # Default: all joints unlocked (checked)
                cb = server.gui.add_checkbox(f"{name}", initial_value=True)
                joint_checkboxes.append(cb)

        @lock_all_btn.on_click
        def _(_):
            for cb in joint_checkboxes:
                if lock_all_btn.value == "Lock":
                    cb.value = False
                elif lock_all_btn.value in ("Unlock", "Reset"):
                    cb.value = True

            if lock_all_btn.value == "Reset":
                reset_requested[0] = True

    # Metrics display.
    with server.gui.add_folder("Metrics"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        pos_error_handle = server.gui.add_number(
            "Position Error (m)", 0.001, disabled=True
        )

    # Get target link index.
    target_link_index = robot.links.names.index(target_link_name)

    # Store previous solution for locked joints.
    default_cfg = np.array(robot.joint_var_cls(0).default_factory())
    prev_cfg = default_cfg.copy()

    while True:
        # Handle reset request.
        if reset_requested[0]:
            prev_cfg = default_cfg.copy()
            reset_requested[0] = False

        # Build joint mask from checkboxes (1.0 = optimize, 0.0 = lock).
        joint_mask = np.array([float(cb.value) for cb in joint_checkboxes])

        # Prepare common arguments.
        target_link_idx_jax = jnp.array(target_link_index, dtype=jnp.int32)
        target_wxyz_jax = jnp.array(ik_target.wxyz)
        target_position_jax = jnp.array(ik_target.position)
        joint_mask_jax = jnp.array(joint_mask)
        prev_cfg_jax = jnp.array(prev_cfg)

        # Solve IK with selected approach.
        start_time = time.time()
        if approach_dropdown.value == "Masked Jacobian":
            solution = _solve_ik_masked(
                robot=robot,
                target_link_index=target_link_idx_jax,
                target_wxyz=target_wxyz_jax,
                target_position=target_position_jax,
                joint_mask=joint_mask_jax,
                prev_cfg=prev_cfg_jax,
            )
        else:
            solution = _solve_ik_retract_masked(
                robot=robot,
                target_link_index=target_link_idx_jax,
                target_wxyz=target_wxyz_jax,
                target_position=target_position_jax,
                joint_mask=joint_mask_jax,
                prev_cfg=prev_cfg_jax,
            )

        # Update timing.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Compute accuracy (position and orientation error).
        pos_err = _compute_pose_error(
            robot=robot,
            joint_cfg=solution,
            target_link_index=target_link_idx_jax,
            target_wxyz=target_wxyz_jax,
            target_position=target_position_jax,
        )
        # Update metrics (EMA smoothing, convert to mm and degrees).
        pos_error_handle.value = (
            0.99 * pos_error_handle.value + 0.01 * float(pos_err) * 1000
        )

        # For locked joints, ensure they stay at the previous value.
        # (The optimizer should produce zero update, but this is a safety net.)
        solution_np = np.array(solution)
        for i, cb in enumerate(joint_checkboxes):
            if not cb.value:  # Joint is locked
                solution_np[i] = prev_cfg[i]

        # Update previous config for next iteration.
        prev_cfg = solution_np.copy()

        # Update visualizer.
        urdf_vis.update_cfg(solution_np)


@jdc.jit
def _compute_pose_error(
    robot: pk.Robot,
    joint_cfg: jax.Array,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    """Compute position and orientation error between actual and target pose.

    Returns:
        pos_error: Position error (Euclidean distance in meters).
    """
    # Compute actual end-effector pose.
    Ts_world_link = robot.forward_kinematics(joint_cfg)
    actual_pose = jaxlie.SE3(Ts_world_link[target_link_index])

    # Compute target pose.
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )

    # Compute pose error in tangent space.
    pose_error = (actual_pose.inverse() @ target_pose).log()

    # Position error (first 3 components of SE3 tangent).
    pos_error = jnp.linalg.norm(pose_error[:3])

    return pos_error


@jdc.jit
def _solve_ik_masked(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    joint_mask: jax.Array,
    prev_cfg: jax.Array,
) -> jax.Array:
    """Solve IK with masked (locked) joints.

    Args:
        robot: Robot model.
        target_link_index: Index of the target link.
        target_wxyz: Target orientation quaternion (w, x, y, z).
        target_position: Target position (x, y, z).
        joint_mask: Array of shape (n_actuated,). 1.0 = optimize, 0.0 = lock.
        prev_cfg: Previous joint configuration (used as initial guess).

    Returns:
        Optimized joint configuration.
    """
    joint_var = robot.joint_var_cls(0)

    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )

    costs = [
        # Use the masked pose cost - locked joints get zero Jacobian.
        pk.costs.pose_cost_analytic_jac_masked(
            robot,
            joint_var,
            target_pose,
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
            joint_mask=joint_mask,
        ),
        # Joint limits (still applied to all joints).
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]),
        )
    )
    return sol[joint_var]


@jdc.jit
def _solve_ik_retract_masked(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    joint_mask: jax.Array,
    prev_cfg: jax.Array,
) -> jax.Array:
    """Solve IK with retract-based joint locking.

    Instead of masking the Jacobian, this approach uses a custom retract_fn
    that filters delta updates for locked joints.

    Args:
        robot: Robot model.
        target_link_index: Index of the target link.
        target_wxyz: Target orientation quaternion (w, x, y, z).
        target_position: Target position (x, y, z).
        joint_mask: Array of shape (n_actuated,). 1.0 = optimize, 0.0 = lock.
        prev_cfg: Previous joint configuration (used as initial guess).

    Returns:
        Optimized joint configuration.
    """

    # Custom retract function that filters delta updates for locked joints.
    def retract_fn(vals: jax.Array, delta: jax.Array) -> jax.Array:
        return vals + delta * joint_mask

    # Create a custom variable class with the masked retract function.
    class MaskedJointVar(
        jaxls.Var[jax.Array],
        default_factory=robot.joint_var_cls(0).default_factory,
        tangent_dim=robot.joints.num_actuated_joints,
        retract_fn=retract_fn,
    ): ...

    joint_var = MaskedJointVar(0)

    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )

    costs = [
        # Use standard pose cost (not masked) - retract handles locking.
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            target_pose,
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        # Joint limits (still applied to all joints).
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]),
        )
    )
    return sol[joint_var]


if __name__ == "__main__":
    main()
