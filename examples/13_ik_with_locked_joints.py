"""IK with Locked Joints

Demonstrates joint locking during IK optimization using the masked pose cost.
Uses checkboxes to dynamically lock/unlock individual joints without
triggering JAX recompilation.
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
    num_joints = robot.joints.num_actuated_joints

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )

    # Create GUI folder for joint locking controls.
    with server.gui.add_folder("Joint Locking"):
        server.gui.add_markdown(
            "Uncheck to lock joints. Uses warm-starting from previous solution."
        )

        # Create checkboxes for each joint (checked = unlocked/optimized).
        joint_checkboxes = []
        for i, name in enumerate(robot.joints.actuated_names):
            # Default: all joints unlocked (checked)
            cb = server.gui.add_checkbox(f"{name}", initial_value=True)
            joint_checkboxes.append(cb)

        # Add button to lock/unlock all.
        lock_all_btn = server.gui.add_button("Lock All")
        unlock_all_btn = server.gui.add_button("Unlock All")

        @lock_all_btn.on_click
        def _(_):
            for cb in joint_checkboxes:
                cb.value = False

        @unlock_all_btn.on_click
        def _(_):
            for cb in joint_checkboxes:
                cb.value = True

    # Reset button to escape bad configurations.
    reset_btn = server.gui.add_button("Reset to Default Pose")
    reset_requested = [False]  # Use list to allow mutation in callback

    @reset_btn.on_click
    def _(_):
        reset_requested[0] = True

    # Timing display.
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

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

        # Solve IK with masked joints.
        start_time = time.time()
        solution = _solve_ik_masked(
            robot=robot,
            target_link_index=jnp.array(target_link_index, dtype=jnp.int32),
            target_wxyz=jnp.array(ik_target.wxyz),
            target_position=jnp.array(ik_target.position),
            joint_mask=jnp.array(joint_mask),
            prev_cfg=jnp.array(prev_cfg),
        )

        # Update timing.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

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


if __name__ == "__main__":
    main()
