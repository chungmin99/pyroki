"""IK with Collision

Basic Inverse Kinematics with Collision Avoidance using PyRoKi.
"""

import json
import time
from pathlib import Path

import numpy as np
import pyroki as pk
import viser
import yourdfpy
from pyroki.collision import HalfSpace, RobotCollision, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description
from trimesh.scene import Scene
from viser.extras import ViserUrdf

import pyroki_snippets as pks


def main():
    """Main function for basic IK with collision."""
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"
    robot = pk.Robot.from_urdf(urdf)

    # Load sphere decomposition from JSON.
    # This was generated through `ballpark` https://github.com/chungmin99/ballpark
    sphere_json_path = Path(__file__).parent / "assets" / "panda_spheres.json"
    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)

    # Create both collision models.
    robot_coll_capsule = RobotCollision.from_urdf(urdf)
    robot_coll_sphere = RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )

    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create interactive controller for IK target.
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )

    # Create interactive controller and mesh for the sphere obstacle.
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # GUI controls for collision type and visualization.
    coll_type_handle = server.gui.add_dropdown(
        "Collision Type", ("Capsule", "Sphere"), initial_value="Capsule"
    )
    show_coll_handle = server.gui.add_checkbox(
        "Show Collision Bodies", initial_value=False
    )

    # Create collision body visualization handles once (attached to ViserUrdf frames).
    # These inherit transforms from urdf_vis.update_cfg() automatically.
    coll_handles_capsule = _create_collision_handles(
        server, urdf, robot_coll_capsule, "capsule", "/robot", (0.0, 1.0, 0.0, 0.5)
    )
    coll_handles_sphere = _create_collision_handles(
        server, urdf, robot_coll_sphere, "sphere", "/robot", (0.0, 1.0, 0.0, 0.5)
    )

    while True:
        start_time = time.time()

        # Select collision model based on dropdown.
        robot_coll = (
            robot_coll_sphere
            if coll_type_handle.value == "Sphere"
            else robot_coll_capsule
        )

        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )

        world_coll_list = [plane_coll, sphere_coll_world_current]
        solution = pks.solve_ik_with_collision(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_position=np.array(ik_target_handle.position),
            target_wxyz=np.array(ik_target_handle.wxyz),
        )

        # Update timing handle.
        timing_handle.value = (time.time() - start_time) * 1000

        # Update visualizer - collision meshes inherit these transforms automatically.
        urdf_vis.update_cfg(solution)

        # Toggle collision body visibility (no mesh regeneration needed).
        show_capsule = show_coll_handle.value and coll_type_handle.value == "Capsule"
        show_sphere = show_coll_handle.value and coll_type_handle.value == "Sphere"
        for h in coll_handles_capsule:
            h.visible = show_capsule
        for h in coll_handles_sphere:
            h.visible = show_sphere


def _get_link_frame_paths(scene: Scene, prefix: str) -> dict[str, str]:
    """Compute viser frame paths for each link in the URDF scene graph.

    ViserUrdf creates hierarchical frames like /robot/visual/link1/link2/...
    This replicates the path computation from viser's _viser_name_from_frame().
    """
    result: dict[str, str] = {}
    base = scene.graph.base_frame
    parents = scene.graph.transforms.parents

    # The base frame itself maps to the prefix (e.g., /robot/visual)
    result[base] = prefix

    # All other frames are children in the hierarchy
    for frame_name in parents.keys():
        frames = []
        current = frame_name
        while current != base:
            frames.append(current)
            current = parents[current]
        frames.append(prefix)
        result[frame_name] = "/".join(frames[::-1])
    return result


def _create_collision_handles(
    server: viser.ViserServer,
    urdf: yourdfpy.URDF,
    robot_coll: RobotCollision,
    prefix: str,
    root_node_name: str = "/robot",
    color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5),
) -> list[viser.SceneNodeHandle]:
    """Create collision mesh handles attached to ViserUrdf's frame hierarchy.

    The meshes are created once and attached as children of the corresponding
    link frames in ViserUrdf. When urdf_vis.update_cfg() is called, the collision
    meshes automatically move with the robot since they inherit the frame transforms.
    """
    handles: list[viser.SceneNodeHandle] = []
    link_paths = _get_link_frame_paths(urdf.scene, f"{root_node_name}/visual")

    for link_name, mesh in robot_coll.get_link_collision_meshes().items():
        if mesh.is_empty or link_name not in link_paths:
            continue
        handle = server.scene.add_mesh_simple(
            f"{link_paths[link_name]}/{prefix}_{link_name}_coll",
            vertices=mesh.vertices.astype(np.float32),
            faces=mesh.faces,
            color=color[:3],
            opacity=color[3],
        )
        handle.visible = False
        handles.append(handle)
    return handles


if __name__ == "__main__":
    main()
