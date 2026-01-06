"""Robot Collision Sweep Visualization

Interactive test script for visualizing swept collision volumes between two robot poses.
Supports both sphere and capsule collision representations with two TransformControl
handles for start and end EE poses.

Usage:
    python tests/viz_collision_sweep.py
"""

from typing import Literal

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import tyro
import viser
from pyroki.collision import RobotCollision
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


# -------------------------------------------------------------------
# IK Solver (copied from pyroki_snippets/_solve_ik.py)
# -------------------------------------------------------------------


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    """JIT-compiled IK solver."""
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """Solve IK for a target end-effector pose."""
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
    )
    return np.array(cfg)


# -------------------------------------------------------------------
# Main Visualization
# -------------------------------------------------------------------


def main(robot_name: Literal["ur5", "panda"] = "panda"):
    """Main function for collision sweep visualization."""
    # Load robot and collision models.
    urdf = load_robot_description(f"{robot_name}_description")
    robot = pk.Robot.from_urdf(urdf)

    if robot_name == "ur5":
        target_link_name = "ee_link"
    elif robot_name == "panda":
        target_link_name = "panda_hand"
    else:
        raise ValueError(f"Unsupported robot name: {robot_name}")

    # Load sphere decomposition from JSON.
    sphere_json_path = (
        Path(__file__).parent.parent
        / "examples"
        / "assets"
        / f"{robot_name}_spheres.json"
    )
    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)

    # Create both collision models.
    robot_coll_capsule = RobotCollision.from_urdf(urdf)
    robot_coll_sphere = RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis_start = ViserUrdf(server, urdf, root_node_name="/robot_start")
    urdf_vis_end = ViserUrdf(server, urdf, root_node_name="/robot_end")

    # Create TransformControl handles for start and end EE poses.
    start_handle = server.scene.add_transform_controls(
        "/ik_target_start",
        scale=0.15,
        position=(0.4, -0.2, 0.5),
        wxyz=(0, 0, 1, 0),
        depth_test=False,
    )
    end_handle = server.scene.add_transform_controls(
        "/ik_target_end",
        scale=0.15,
        position=(0.4, 0.2, 0.5),
        wxyz=(0, 0, 1, 0),
        depth_test=False,
    )

    # Add frame markers at the control positions.
    server.scene.add_frame(
        "/ik_target_start/frame",
        axes_length=0.05,
        axes_radius=0.005,
    )
    server.scene.add_frame(
        "/ik_target_end/frame",
        axes_length=0.05,
        axes_radius=0.005,
    )

    # GUI controls.
    coll_type_handle = server.gui.add_dropdown(
        "Collision Type", ("Capsule", "Sphere"), initial_value="Capsule"
    )
    show_swept_handle = server.gui.add_checkbox("Show Swept Volume", initial_value=True)
    opacity_handle = server.gui.add_slider(
        "Swept Opacity", min=0.1, max=1.0, step=0.1, initial_value=0.5
    )
    timing_handle = server.gui.add_number("IK Time (ms)", 0.001, disabled=True)

    # State tracking.
    last_start_pose: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    last_end_pose: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    last_coll_type: str | None = None
    last_opacity: float | None = None
    swept_handle: viser.SceneNodeHandle | None = None
    cfg_start: np.ndarray | None = None
    cfg_end: np.ndarray | None = None

    print("Collision Sweep Visualization running at http://localhost:8080")

    # Perform initial IK solve to show swept volume at startup.
    cfg_start = solve_ik(
        robot,
        target_link_name,
        np.array(start_handle.wxyz),
        np.array(start_handle.position),
    )
    cfg_end = solve_ik(
        robot,
        target_link_name,
        np.array(end_handle.wxyz),
        np.array(end_handle.position),
    )
    urdf_vis_start.update_cfg(cfg_start)
    urdf_vis_end.update_cfg(cfg_end)

    while True:
        # Get current state.
        current_start = (tuple(start_handle.position), tuple(start_handle.wxyz))
        current_end = (tuple(end_handle.position), tuple(end_handle.wxyz))
        current_coll_type = coll_type_handle.value
        current_opacity = opacity_handle.value

        # Check if anything changed.
        poses_changed = current_start != last_start_pose or current_end != last_end_pose
        coll_type_changed = current_coll_type != last_coll_type
        opacity_changed = current_opacity != last_opacity

        needs_rebuild = poses_changed or coll_type_changed or opacity_changed

        if poses_changed:
            start_time = time.time()

            # Solve IK for both poses.
            cfg_start = solve_ik(
                robot,
                target_link_name,
                np.array(start_handle.wxyz),
                np.array(start_handle.position),
            )
            cfg_end = solve_ik(
                robot,
                target_link_name,
                np.array(end_handle.wxyz),
                np.array(end_handle.position),
            )

            timing_handle.value = (time.time() - start_time) * 1000

            # Update robot visualization to show end configuration.
            urdf_vis_start.update_cfg(cfg_start)
            urdf_vis_end.update_cfg(cfg_end)

            last_start_pose = current_start
            last_end_pose = current_end

        if needs_rebuild:
            # Select collision model.
            robot_coll = (
                robot_coll_sphere
                if current_coll_type == "Sphere"
                else robot_coll_capsule
            )

            # Compute swept capsules (convert to JAX arrays).
            swept_caps = robot_coll.get_swept_capsules(
                robot, jnp.array(cfg_start), jnp.array(cfg_end)
            )

            # Remove old mesh.
            if swept_handle is not None:
                swept_handle.remove()
                swept_handle = None

            # Create new swept volume mesh.
            if show_swept_handle.value:
                mesh = swept_caps.to_trimesh()
                swept_handle = server.scene.add_mesh_simple(
                    "/swept_volume",
                    vertices=mesh.vertices.astype(np.float32),
                    faces=mesh.faces,
                    color=(0.2, 0.8, 0.2),
                    opacity=current_opacity,
                )

            last_coll_type = current_coll_type
            last_opacity = current_opacity

        # Handle visibility toggle.
        if swept_handle is not None:
            swept_handle.visible = show_swept_handle.value

        time.sleep(0.01)


if __name__ == "__main__":
    tyro.cli(main)
