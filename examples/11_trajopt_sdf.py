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
    def make_grid(voxel, dims_xyz):
        nx, ny, nz = dims_xyz
        dx, dy, dz = voxel
        xs = (jnp.arange(nx) - (nx - 1) / 2) * dx
        ys = (jnp.arange(ny) - (ny - 1) / 2) * dy
        zs = (jnp.arange(nz) - (nz - 1) / 2) * dz
        grid_pts = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="xy"), axis=-1)
        grid_pts = jnp.moveaxis(grid_pts, 2, 0)     # (z,y,x,3)
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
    voxel = jnp.array([0.02, 0.02, 0.02])
    dims_xyz = (64, 64, 64)
    idx_half = (jnp.array(dims_xyz) - 1) / 2.0            # (31.5, 31.5, 31.5)
    sphere_w = jnp.array([0.5, 0.0, 0.18])                 # where you want the centre

    # 1. Build a *corner-anchored* grid (origin at (0,0,0) local)
    def make_grid_corner(voxel, dims_xyz):
        nx, ny, nz = dims_xyz
        dx, dy, dz = voxel
        xs = jnp.arange(nx) * dx
        ys = jnp.arange(ny) * dy
        zs = jnp.arange(nz) * dz
        pts = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="xy"), axis=-1)
        pts = jnp.moveaxis(pts, 2, 0)           # (z,y,x,3)
        return pts

    grid_pts = make_grid_corner(voxel, dims_xyz)

    # 2. Build sphere SDF in *local* coordinates
    sphere_sdf = sdf_sphere(grid_pts,
                            centre=idx_half * voxel,   #  centre at array middle
                            radius=0.2)

    # 3. Place the grid so that index (0,0,0) sits at world = sphere_w − idx_half*voxel
    grid_origin_w = sphere_w - idx_half * voxel

    grid = pk.collision.SDFGrid(
        pose       = jaxlie.SE3.from_translation(grid_origin_w),   # CORNER
        voxel_size = voxel,
        size       = voxel,
        sdf        = sphere_sdf,
    )


    box_sdf    = sdf_box(grid_pts,
                        centre=jnp.array([0.3, 0.0, 0.10]),
                        half_extents=jnp.array([0.08, 0.30, 0.10]))

    # *union* (can also take min() for intersection, -min(-d) for union of negatives, etc.)
    combined_sdf = jnp.minimum(sphere_sdf, box_sdf)      # (z,y,x)
    combined_sdf = sphere_sdf

    # ---------------------------------------------------------------------
    # Wrap it in an SDFGrid CollGeom (see previous answer §1)
    # ---------------------------------------------------------------------

    # grid = pk.collision.SDFGrid(
    #     pose = jaxlie.SE3.from_translation(jnp.array([0.7, 0.0, 0.0])),  # centre of sphere
    #     size = voxel,
    #     voxel_size = voxel,
    #     sdf  = combined_sdf,
    # )

    # Now `grid_geom` can be appended to your `world_coll` list:
    # world_coll = [ground_coll, grid_geom]
    world_coll = [ground_coll, grid]    # ← feed into solve_trajopt exactly like before

    import jax.numpy as jnp
    from functools import partial

    # -------------------------------------------------------
    # 1. zero-level sanity check
    # -------------------------------------------------------
    sphere_center_w = jnp.array([0.7, 0.0, 0.2])          # same as you used
    # single-point query – robust to any  leading broadcast axes
    sdf_at_center = grid._interpolate_sdf(sphere_center_w).reshape(()).item()
    print(f"SDF(center) = {sdf_at_center:+.4f} m")

    print(f"SDF(center)   = {sdf_at_center:+.4f} m "
        "(should be ~0: inside the grid & oriented right)")
    
    sdf_at_center = grid._interpolate_sdf(sphere_w).reshape(()).item()
    print(f"SDF(center) = {sdf_at_center:+.4f}  (≈ -0.30 expected)")

    # -------------------------------------------------------
    # 2. grid pose & extent sanity check
    # -------------------------------------------------------
    grid_min_w = grid.pose.apply(jnp.array([0, 0, 0]))                 # corner
    grid_max_w = grid.pose.apply(
        (jnp.array(grid.sdf.shape[::-1]) - 1) * grid.voxel_size)       # opposite corner
    print(f"Grid spans X:[{grid_min_w[0]:+.2f},{grid_max_w[0]:+.2f}] m "
        f"Y:[{grid_min_w[1]:+.2f},{grid_max_w[1]:+.2f}] m "
        f"Z:[{grid_min_w[2]:+.2f},{grid_max_w[2]:+.2f}] m")


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

    import jax.numpy as jnp
    from functools import partial

    # # Wrap once so we JIT only a single function call, not 25×
    # @partial(jax.jit, static_argnums=(0, 1))
    def min_signed_distance(robot, robot_coll, cfg, world_geom):
        dist = robot_coll.compute_world_collision_distance(robot, cfg, world_geom)
        return jnp.min(dist)          # single scalar: most negative penetration

    for t, q in enumerate(traj):
        d = float(min_signed_distance(robot, robot_coll,
                                    jnp.asarray(q),   # cfg shape (DoF,)
                                    grid))            # or world_coll[1]
        print(f"step {t:02d}  min-dist = {d:+.4f} m")


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

    # # ---- SDF  → trimesh  --------------------------------------------------
    # # import numpy as np
    # # import trimesh
    # from trimesh.voxel import ops as voxel_ops     # uses skimage’s marching-cubes

    # # jax → numpy
    # sdf_np   = jnp.asarray(combined_sdf)         # (Z, Y, X)
    # inside   = sdf_np <= 0.0                    # boolean occupancy

    # # trimesh expects (X, Y, Z) ordering
    # mc_mesh = voxel_ops.matrix_to_marching_cubes(
    #     inside.transpose(2, 1, 0),              # → (X,Y,Z)
    #     pitch=float(voxel[0])                   # or tuple(voxel) if anisotropic
    # )

    # # move mesh from grid-local coordinates into world coordinates
    # # 1. Evaluate the property           ↓ parentheses!
    # world_offset = np.asarray(origin_local) + np.asarray(grid.pose.translation())

    # mc_mesh.apply_translation(world_offset)


    # # ---- drop into viser --------------------------------------------------
    # server.scene.add_mesh_trimesh(
    #     name="/sdf_mesh",
    #     mesh=mc_mesh,
    #     wxyz=(1.0, 0.0, 0.0, 0.0),              # no rotation
    #     visible=True,
    # )


    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps

        urdf_vis.update_cfg(traj[slider.value])
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
