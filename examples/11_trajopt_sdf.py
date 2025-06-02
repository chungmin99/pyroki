"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

import time
from typing import Literal

import numpy as np
import pyroki as pk
import trimesh
from trimesh.voxel import ops as voxel_ops          # marching-cubes helper
import tyro
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks

import jax.numpy as jnp
import jaxlie

def make_sphere_grid(world_center: jnp.ndarray,
                     radius: float,
                     voxel: tuple[float, float, float] = (0.02, 0.02, 0.02),
                     dims: tuple[int, int, int] = (64, 64, 64),
) -> pk.collision.SDFGrid:
    """
    Build a corner-anchored cubic SDF grid containing a single sphere.

    Parameters
    ----------
    world_center : (3,) world-frame xyz of the sphere centre.
    radius       : sphere radius [m].
    voxel        : (dx, dy, dz) voxel size [m].
    dims         : (nx, ny, nz) number of cells.

    Returns
    -------
    SDFGrid  ready to append to `world_coll`.
    """
    voxel = jnp.asarray(voxel)
    nx, ny, nz = dims
    # local voxel centres (corner-anchored)
    xs = jnp.arange(nx) * voxel[0]
    ys = jnp.arange(ny) * voxel[1]
    zs = jnp.arange(nz) * voxel[2]
    pts = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="xy"), axis=-1)
    pts = jnp.moveaxis(pts, 2, 0)                     # (z,y,x,3)

    idx_half = (jnp.asarray(dims) - 1) / 2            # 31.5,31.5,31.5
    local_center = idx_half * voxel                   # metres in grid frame

    sdf = jnp.linalg.norm(pts - local_center, axis=-1) - radius

    grid_origin_w = world_center - local_center       # place array corner
    pose = jaxlie.SE3.from_translation(grid_origin_w)

    return pk.collision.SDFGrid(
        pose       = pose,
        voxel_size = voxel,
        size       = voxel,   # not used in distance but kept for completeness
        sdf        = sdf,
    )


def add_sdf_to_viser(server, name, grid, level=0.0):
    # ------------ 1. extract mesh in grid frame --------------------
    sdf_np = np.array(grid.sdf, dtype=np.float32)        # make *writable*
    sdf_np = sdf_np.transpose(2, 1, 0)                   # (X,Y,Z) for both libs

    verts, faces = None, None
    try:
        # prefer skimage if available
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(
            sdf_np, level=level, spacing=tuple(grid.voxel_size[::-1]))
    except Exception:
        # occupancy fallback (dx must equal dy == dz)
        inside = sdf_np <= level
        verts, faces = voxel_ops.matrix_to_marching_cubes(
            inside, pitch=float(grid.voxel_size[0]))
        verts *= np.asarray(grid.voxel_size[::-1])        # rescale anisotropic

    mesh = trimesh.Trimesh(verts, faces, process=False)

    # ------------ 2. move to world frame --------------------------
    mesh.apply_transform(grid.pose.as_matrix())

    # ------------ 3. send to Viser -------------------------------
    server.scene.add_mesh_trimesh(name=name, mesh=mesh)


def main(robot_name: Literal["ur5", "panda"] = "panda"):
    if robot_name == "ur5":
        urdf = load_robot_description("ur5_description")
        down_wxyz = np.array([0.707, 0, 0.707, 0])
        target_link_name = "ee_link"

        # For UR5 it's important to initialize the robot in a safe configuration;
        # the zero-configuration puts the robot aligned with the wall obstacle.
        default_cfg = np.zeros(6)
        default_cfg[1] = -1.308
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)

    elif robot_name == "panda":
        urdf = load_robot_description("panda_description")
        target_link_name = "panda_hand"
        down_wxyz = np.array([0, 0, 1, 0])  # for panda!
        robot = pk.Robot.from_urdf(urdf)

    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # Define the trajectory problem:
    # - number of timesteps, timestep size
    timesteps, dt = 25, 0.02
    # - the start and end poses.
    start_pos, end_pos = np.array([0.5, -0.3, 0.2]), np.array([0.5, 0.3, 0.2])

    # Define the obstacles:
    # - Ground
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    # - Wall
    wall_height = 0.4
    wall_width = 0.1
    wall_length = 0.4
    wall_intervals = np.arange(start=0.3, stop=wall_length + 0.3, step=0.05)
    translation = np.concatenate(
        [
            wall_intervals.reshape(-1, 1),
            np.full((wall_intervals.shape[0], 1), 0.0),
            np.full((wall_intervals.shape[0], 1), wall_height / 2),
        ],
        axis=1,
    )

    grid = make_sphere_grid(world_center=jnp.array([0.5, 0.0, 0.18]),
                        radius=0.20)
    world_coll = [ground_coll, grid]

    traj = pks.solve_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_name,
        start_pos,
        down_wxyz,
        end_pos,
        down_wxyz,
        timesteps,
        dt,
    )
    traj = np.array(traj)

    # Visualize!
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf)
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
    server.scene.add_mesh_trimesh(
        "wall_box",
        trimesh.creation.box(
            extents=(wall_length, wall_width, wall_height),
            transform=trimesh.transformations.translation_matrix(
                np.array([0.5, 0.0, wall_height / 2])
            ),
        ),
    )
    for name, pos in zip(["start", "end"], [start_pos, end_pos]):
        server.scene.add_frame(
            f"/{name}",
            position=pos,
            wxyz=down_wxyz,
            axes_length=0.05,
            axes_radius=0.01,
        )

    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    add_sdf_to_viser(server, "/sdf_mesh", grid)


    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps

        urdf_vis.update_cfg(traj[slider.value])
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
