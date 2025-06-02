"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

import time
from typing import Literal

import numpy as np
import pyroki as pk
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks


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
    # wall_coll = pk.collision.Capsule.from_radius_height(
    #     position=translation,
    #     radius=np.full((translation.shape[0], 1), wall_width / 2),
    #     height=np.full((translation.shape[0], 1), wall_height),
    # )
    # world_coll = [ground_coll, wall_coll]

    import jax.numpy as jnp
    import jax
    import trimesh                     # only for quick visual check (optional)
    import jaxlie

    # ---------------------------------------------------------------------
    # Helper: make a voxel grid centred at `origin` with resolution `voxel`
    # ---------------------------------------------------------------------
    def make_grid(origin_xyz, voxel, dims_xyz):
        """
        origin_xyz : (3,)  world-frame coordinates of grid corner (x0,y0,z0)
        voxel      : (3,)  voxel sizes (dx,dy,dz)
        dims_xyz   : (3,)  integer number of cells in x,y,z
        """
        nx, ny, nz = dims_xyz
        dx, dy, dz = voxel
        # coordinate centres of every voxel in world frame
        xs = origin_xyz[0] + (jnp.arange(nx) + 0.5) * dx
        ys = origin_xyz[1] + (jnp.arange(ny) + 0.5) * dy
        zs = origin_xyz[2] + (jnp.arange(nz) + 0.5) * dz
        # shape (nz,ny,nx,3) –– we keep z-major so marching-cubes later is easier
        grid_pts = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="xy"), axis=-1)
        grid_pts = jnp.moveaxis(grid_pts, 2, 0)              # (z,y,x,3)
        return grid_pts


    # ---------------------------------------------------------------------
    # Primitives: sphere SDF and axis-aligned box SDF
    # ---------------------------------------------------------------------
    def sdf_sphere(pts, centre, radius):
        """Signed distance to a sphere."""
        return jnp.linalg.norm(pts - centre, axis=-1) - radius

    def sdf_box(pts, centre, half_extents):
        """Axis-aligned box; positive outside, negative inside."""
        q = jnp.abs(pts - centre) - half_extents
        outside = jnp.linalg.norm(jnp.clip(q, 0), axis=-1)
        inside  = jnp.max(jnp.minimum(q, 0), axis=-1)
        return outside + inside


    # ---------------------------------------------------------------------
    # Build a 64³ grid with two obstacles
    # ---------------------------------------------------------------------
    voxel       = jnp.array([0.02, 0.02, 0.02])          # 2 cm voxels
    dims_xyz    = (64, 64, 64)
    origin_xyz  = jnp.array([-0.64, -0.64, -0.02])       # put z=0 roughly in the middle

    grid_pts = make_grid(origin_xyz, voxel, dims_xyz)    # (64,64,64,3)
    sphere_sdf = sdf_sphere(grid_pts,
                            centre=jnp.array([0.0, 0.0, 0.25]),
                            radius=0.50)

    box_sdf    = sdf_box(grid_pts,
                        centre=jnp.array([0.3, 0.0, 0.10]),
                        half_extents=jnp.array([0.08, 0.30, 0.10]))

    # *union* (can also take min() for intersection, -min(-d) for union of negatives, etc.)
    combined_sdf = jnp.minimum(sphere_sdf, box_sdf)      # (z,y,x)
    combined_sdf = jnp.minimum(sphere_sdf)

    # ---------------------------------------------------------------------
    # Wrap it in an SDFGrid CollGeom (see previous answer §1)
    # ---------------------------------------------------------------------

    grid = pk.collision.SDFGrid(
        pose        = jaxlie.SE3.identity(),             # grid frame ≡ world for now
        size        = voxel,                             # optional; not used for distance
        sdf         = combined_sdf,                      # (Z,Y,X)
        voxel_size  = voxel,
    )

    # Now `grid_geom` can be appended to your `world_coll` list:
    # world_coll = [ground_coll, grid_geom]
    world_coll = [ground_coll, grid]    # ← feed into solve_trajopt exactly like before


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

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps

        urdf_vis.update_cfg(traj[slider.value])
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
