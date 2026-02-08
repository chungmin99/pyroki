"""J-PARSE Velocity IK

Singularity-aware velocity IK using J-PARSE with manipulability visualization.
"""

import time

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from pyroki_snippets._jparse import jparse_step


def main():
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

    # GUI controls.
    gamma_handle = server.gui.add_slider(
        "Gamma (singularity threshold)", 0.01, 0.5, 0.01, 0.1
    )
    nullspace_gain_handle = server.gui.add_slider("Nullspace gain", 0.0, 2.0, 0.05, 0.5)
    manipulability_handle = server.gui.add_number(
        "Manipulability", 0.001, disabled=True
    )
    show_ellipsoid_handle = server.gui.add_checkbox("Show ellipsoid", True)

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
        cfg, info = jparse_step(
            robot=robot,
            cfg=cfg,
            target_link_index=target_link_index,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
            gamma=gamma_handle.value,
            nullspace_gain=nullspace_gain_handle.value,
            dt=0.02,
        )

        # Update metrics and visualization.
        manipulability_handle.value = round(info["manipulability"], 4)
        manip_ellipse.set_visibility(show_ellipsoid_handle.value)
        manip_ellipse.update(cfg)
        urdf_vis.update_cfg(cfg)

        # Sleep to maintain ~50Hz.
        elapsed = time.time() - start_time
        time.sleep(max(0.0, 0.02 - elapsed))


if __name__ == "__main__":
    main()
