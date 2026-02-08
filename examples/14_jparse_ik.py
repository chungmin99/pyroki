"""J-PARSE Velocity IK

Singularity-aware velocity IK using J-PARSE with PyRoKi.

J-PARSE (Jacobian-based Projection Algorithm for Resolving Singularities
Effectively) provides smooth behavior near singularities by decomposing
the Jacobian into non-singular and singular directions.

Compare J-PARSE against standard pseudo-inverse and damped least squares
methods using the GUI controls.
"""

import time

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from pyroki_snippets._jparse import jparse_step


def main():
    """Main function for J-PARSE velocity IK example."""

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)
    target_link_index = robot.links.names.index(target_link_name)

    # Initial configuration (middle of joint range).
    cfg = np.array((robot.joints.lower_limits + robot.joints.upper_limits) / 2.0)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Target gizmo.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )

    # --- GUI controls ---
    with server.gui.add_folder("Method"):
        method_handle = server.gui.add_dropdown(
            "IK Method",
            options=["jparse", "pinv", "dls"],
            initial_value="jparse",
        )
        gamma_handle = server.gui.add_slider(
            "Gamma (singularity threshold)", 0.01, 0.5, 0.01, 0.1
        )
        singular_gain_position_handle = server.gui.add_slider(
            "Singular gain (position)", 0.5, 5.0, 0.1, 2.0
        )
        singular_gain_angular_handle = server.gui.add_slider(
            "Singular gain (angular)", 0.5, 5.0, 0.1, 2.0
        )

    with server.gui.add_folder("Control"):
        track_orientation_handle = server.gui.add_checkbox("Track orientation", True)
        position_gain_handle = server.gui.add_slider(
            "Position gain", 1.0, 20.0, 0.5, 5.0
        )
        orientation_gain_handle = server.gui.add_slider(
            "Orientation gain", 0.1, 5.0, 0.1, 1.0
        )
        nullspace_gain_handle = server.gui.add_slider(
            "Nullspace gain", 0.0, 2.0, 0.05, 0.5
        )
        max_vel_handle = server.gui.add_slider("Max joint velocity", 0.5, 5.0, 0.1, 2.0)

    with server.gui.add_folder("Visualization"):
        show_ellipsoid_handle = server.gui.add_checkbox("Show ellipsoid", True)

    with server.gui.add_folder("Metrics"):
        manipulability_handle = server.gui.add_number(
            "Manipulability", 0.0, disabled=True
        )
        condition_handle = server.gui.add_number(
            "Inv. condition number", 0.0, disabled=True
        )
        max_vel_display = server.gui.add_number(
            "Max joint vel (rad/s)", 0.0, disabled=True
        )
        pos_error_handle = server.gui.add_number(
            "Position error (m)", 0.0, disabled=True
        )
        ori_error_handle = server.gui.add_number(
            "Orientation error (rad)", 0.0, disabled=True
        )
        singularity_handle = server.gui.add_text(
            "Singularity status", "", disabled=True
        )

    # Manipulability ellipsoid visualization.
    manip_ellipse = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability",
        target_link_name=target_link_name,
    )

    # Control loop at ~50Hz.
    while True:
        start_time = time.time()

        # Take one velocity IK step.
        target_wxyz = (
            np.array(ik_target.wxyz) if track_orientation_handle.value else None
        )
        cfg, info = jparse_step(
            robot=robot,
            cfg=cfg,
            target_link_index=target_link_index,
            target_position=np.array(ik_target.position),
            target_wxyz=target_wxyz,
            method=method_handle.value,  # type: ignore
            gamma=gamma_handle.value,
            singular_direction_gain_position=singular_gain_position_handle.value,
            singular_direction_gain_angular=singular_gain_angular_handle.value,
            position_gain=position_gain_handle.value,
            orientation_gain=orientation_gain_handle.value,
            nullspace_gain=nullspace_gain_handle.value,
            max_joint_velocity=max_vel_handle.value,
            dt=0.02,
        )

        # Update metrics.
        manipulability_handle.value = round(info["manipulability"], 4)
        condition_handle.value = round(info["inverse_condition_number"], 4)
        max_vel_display.value = round(info["max_joint_vel"], 3)
        pos_error_handle.value = round(info["position_error"], 4)
        ori_error_handle.value = round(info["orientation_error"], 4)

        # Singularity warning.
        icn = info["inverse_condition_number"]
        if icn < gamma_handle.value:
            singularity_handle.value = f"NEAR SINGULAR (icn={icn:.3f})"
        else:
            singularity_handle.value = "OK"

        # Update ellipsoid.
        manip_ellipse.set_visibility(show_ellipsoid_handle.value)
        manip_ellipse.update(cfg)

        # Update robot visualization.
        urdf_vis.update_cfg(cfg)

        # Sleep to maintain ~50Hz.
        elapsed = time.time() - start_time
        time.sleep(max(0.0, 0.02 - elapsed))


if __name__ == "__main__":
    main()
