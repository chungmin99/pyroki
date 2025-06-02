"""Online Planning

Run online planning in collision aware environments.
"""

import time

import numpy as np
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import viser
from pyroki.collision import HalfSpace, RobotCollision, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import trimesh
from trimesh.voxel import ops as voxel_ops          # marching-cubes helper

import pyroki_snippets as pks

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

def main():
    """Main function for online planning with collision."""
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"
    robot = pk.Robot.from_urdf(urdf)

    robot_coll = RobotCollision.from_urdf(urdf)
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Define the online planning parameters.
    len_traj, dt = 5, 0.1

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create interactive controller for IK target.
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.3, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )

    # Create interactive controller and mesh for the sphere obstacle.
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())
    target_frame_handle = server.scene.add_batched_axes(
        "target_frame",
        axes_length=0.05,
        axes_radius=0.005,
        batched_positions=np.zeros((25, 3)),
        batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * 25),
    )

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    sol_pos, sol_wxyz = None, None
    sol_traj = np.array(
        robot.joint_var_cls.default_factory()[None].repeat(len_traj, axis=0)
    )



    grid = make_sphere_grid(world_center=jnp.array([0.5, 0.0, 0.18]),
                        radius=0.20)

    add_sdf_to_viser(server, "/sdf_mesh", grid)

    while True:
        start_time = time.time()

        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )

        world_coll_list = [plane_coll, sphere_coll_world_current, grid]
        sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning(
            robot=robot,
            robot_coll=robot_coll,
            world_coll=world_coll_list,
            target_link_name=target_link_name,
            target_position=np.array(ik_target_handle.position),
            target_wxyz=np.array(ik_target_handle.wxyz),
            timesteps=len_traj,
            dt=dt,
            start_cfg=sol_traj[0],
            prev_sols=sol_traj,
        )

        # Update timing handle.
        timing_handle.value = (
            0.99 * timing_handle.value + 0.01 * (time.time() - start_time) * 1000
        )

        # Update visualizer.
        urdf_vis.update_cfg(
            sol_traj[0]
        )  # The first step of the online trajectory solution.

        # Update the planned trajectory visualization.
        if hasattr(target_frame_handle, "batched_positions"):
            target_frame_handle.batched_positions = np.array(sol_pos)  # type: ignore[attr-defined]
            target_frame_handle.batched_wxyzs = np.array(sol_wxyz)  # type: ignore[attr-defined]
        else:
            # This is an older version of Viser.
            target_frame_handle.positions_batched = np.array(sol_pos)  # type: ignore[attr-defined]
            target_frame_handle.wxyzs_batched = np.array(sol_wxyz)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
