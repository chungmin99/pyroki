"""Climbing Retargeting (Trajectory Optimization)

Retarget motion to G1 humanoid using Laplacian mesh deformation with trajectory optimization.
All timesteps are optimized simultaneously (not sequentially like diffIK).
Based on the holosoma interaction mesh retargeting approach.
"""

import json
import time
from pathlib import Path
from typing import TypedDict

import trimesh
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from retarget_helpers._climb_utils import (
    adjacency_list_to_matrix,
    calculate_laplacian_coordinates_vectorized,
    create_interaction_mesh,
    get_adjacency_list,
    get_climb_retarget_indices,
    load_climb_motion,
    load_object_mesh,
    load_object_points,
)


class ClimbingRetargetingWeights(TypedDict):
    laplacian: float
    """Laplacian mesh deformation weight (default: 10.0 from holosoma)."""
    joint_smoothness: float
    """Smoothness weight towards previous frame (default: 0.2 from holosoma)."""
    pose_smoothness: float
    """Smoothness weight towards previous frame (default: 0.2 from holosoma)."""
    rest: float
    """Rest pose regularization weight (very small, default: 0.01)."""
    self_collision: float
    """Self-collision weight."""
    scale_init: float
    """Initial scale factor for mocap-to-robot scaling (default: 0.742)."""
    scale_reg: float
    """Scale regularization weight to keep scale near initial value."""


# Scale factor optimization variable
class ScaleVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.742)):
    """Optimization variable for mocap-to-robot scale factor."""

    ...


def main():
    """Main function for climbing retargeting."""

    # Load robot.
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    sphere_json_path = Path(__file__).parent / "assets" / "g1_spheres.json"
    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)
    robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )

    # Load motion data and object (unscaled for optimization)
    asset_dir = Path(__file__).parent / "retarget_helpers" / "omniretarget_climb_data"
    motion_unscaled = load_climb_motion(
        asset_dir / "mocap_climb_seq_0_joint_positions_f900-3700.npy",
        scale_factor=1.0,  # Load unscaled - scale will be optimized
    )
    object_points_unscaled = load_object_points(
        asset_dir / "multi_boxes.obj", sample_count=30, scale_factor=1.0
    )
    # Load collision mesh unscaled - will be scaled dynamically by var_scale
    object_mesh_unscaled = load_object_mesh(
        asset_dir / "multi_boxes.obj", scale_factor=1.0
    )
    # Keep a scaled version for initial visualization
    object_mesh_vis = load_object_mesh(
        asset_dir / "multi_boxes.obj", scale_factor=0.742
    )

    # Subsample trajectory for faster optimization
    subsample_factor = 1
    motion_unscaled = motion_unscaled[::subsample_factor]

    num_timesteps = motion_unscaled.shape[0]
    print(
        f"Loaded motion with {num_timesteps} timesteps (subsampled {subsample_factor}x), {motion_unscaled.shape[1]} joints"
    )
    print(f"Object points shape: {object_points_unscaled.shape}")

    # Get retarget indices and local offsets.
    mocap_indices, g1_link_indices, local_offsets = get_climb_retarget_indices(robot)
    print(f"Retargeting {len(mocap_indices)} joints")

    # Setup visualization
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True, disabled=True)
    timestep_slider = server.gui.add_slider(
        "timestep", 0, num_timesteps - 1, 1, 0, disabled=True
    )

    # Add object mesh to scene (use scaled version for initial visualization)
    server.scene.add_mesh_trimesh("/object", object_mesh_vis)

    # Split up the unscaled object mesh into individual boxes for collision detection.
    # These will be scaled dynamically by var_scale during optimization.
    meshes = object_mesh_unscaled.split()
    assert len(meshes) == 3
    box_1_transform, box_1_extents = trimesh.bounds.oriented_bounds(meshes[0])
    box_2_transform, box_2_extents = trimesh.bounds.oriented_bounds(meshes[1])
    box_3_transform, box_3_extents = trimesh.bounds.oriented_bounds(meshes[2])
    box_1_jaxlie_transform = jaxlie.SE3.from_matrix(box_1_transform).inverse()
    box_2_jaxlie_transform = jaxlie.SE3.from_matrix(box_2_transform).inverse()
    box_3_jaxlie_transform = jaxlie.SE3.from_matrix(box_3_transform).inverse()

    coll_box = pk.collision.Box.from_extent(
        extent=onp.array([box_1_extents, box_2_extents, box_3_extents]),
        position=onp.array(
            [
                box_1_jaxlie_transform.translation(),
                box_2_jaxlie_transform.translation(),
                box_3_jaxlie_transform.translation(),
            ]
        ),
        wxyz=onp.array(
            [
                box_1_jaxlie_transform.rotation().wxyz,
                box_2_jaxlie_transform.rotation().wxyz,
                box_3_jaxlie_transform.rotation().wxyz,
            ]
        ),
    )
    coll_plane = pk.collision.HalfSpace.from_point_and_normal(
        point=jnp.array([0.0, 0.0, 0.0]),
        normal=jnp.array([0.0, 0.0, 1.0]),
    )
    server.scene.add_mesh_trimesh(
        "/box_coll_pk", coll_box.to_trimesh(), receive_shadow=0.1
    )

    # Add ground plane
    server.scene.add_grid("/grid", width=4, height=4, position=(0.0, 0.0, 0.0))

    weights = pk.viewer.WeightTuner(
        server,
        ClimbingRetargetingWeights(
            laplacian=3.0,
            joint_smoothness=1.0,
            pose_smoothness=1.0,
            rest=0.1,
            self_collision=0.5,
            scale_init=1.0,  # Initial scale factor
            scale_reg=0.1,  # Scale regularization weight
        ),  # type: ignore
    )
    rerun_button = server.gui.add_button("Retarget Trajectory")
    scale_label = server.gui.add_text(
        "Optimized scale", initial_value="--", disabled=True
    )

    # Precompute interaction mesh (must be done outside JIT)
    n_robot_pts = len(mocap_indices)
    n_obj_pts = object_points_unscaled.shape[0]
    n_total_pts = n_robot_pts + n_obj_pts

    # Build interaction mesh from first frame (using unscaled data)
    demo_robot_pts = motion_unscaled[0, mocap_indices]
    all_vertices_np = onp.concatenate(
        [onp.array(demo_robot_pts), onp.array(object_points_unscaled)], axis=0
    )
    _, tetrahedra = create_interaction_mesh(all_vertices_np)
    adj_list = get_adjacency_list(tetrahedra, n_total_pts)
    adj_matrix = adjacency_list_to_matrix(adj_list, n_total_pts)

    # Precompute target Laplacian coordinates for each frame (unscaled)
    def compute_target_lap(keypoints_frame: jnp.ndarray) -> jnp.ndarray:
        robot_pts = keypoints_frame[mocap_indices]
        all_pts = jnp.concatenate([robot_pts, object_points_unscaled], axis=0)
        return calculate_laplacian_coordinates_vectorized(all_pts, adj_matrix)

    target_laplacians_unscaled = jax.vmap(compute_target_lap)(motion_unscaled)
    assert target_laplacians_unscaled.shape == (num_timesteps, n_total_pts, 3)

    # Get the current robot status, step-by-step.
    list_T_world_root, list_joints = [], []
    current_scale = 0.742  # Will be updated by optimization

    def retarget_trajectory():
        nonlocal list_T_world_root, list_joints, current_scale

        with server.atomic():
            # Temporarily disable weight tuning during solve.
            for handle in weights._weight_handles.values():
                handle.disabled = True
            rerun_button.disabled = True

        # Compute warm-start initialization via sequential IK on key frames
        init_Ts_wxyz_xyz, init_joints_warmstart, init_scale = (
            compute_warmstart_initialization(
                robot=robot,
                robot_coll=robot_coll,
                world_coll_list=[coll_box, coll_plane],
                target_laplacians_unscaled=target_laplacians_unscaled,
                object_points_unscaled=object_points_unscaled,
                g1_link_indices=g1_link_indices,
                local_offsets=local_offsets,
                adj_matrix=adj_matrix,
                weights=weights.get_weights(),  # type: ignore
                motion_unscaled=motion_unscaled,
                mocap_indices=mocap_indices,
                keyframe_interval=12,  # Reduced from 50 due to 4x subsampling
            )
        )

        print("Solving trajectory optimization (all timesteps simultaneously)...")

        # Single trajectory optimization call
        Ts_world_root, joints, final_scale = solve_trajectory(
            robot=robot,
            robot_coll=robot_coll,
            world_coll_list=[coll_box, coll_plane],
            target_laplacians_unscaled=target_laplacians_unscaled,
            object_points_unscaled=object_points_unscaled,
            g1_link_indices=g1_link_indices,
            local_offsets=local_offsets,
            adj_matrix=adj_matrix,
            weights=weights.get_weights(),  # type: ignore
            init_Ts_wxyz_xyz=init_Ts_wxyz_xyz,
            init_joints=init_joints_warmstart,
            init_scale=init_scale,
        )
        jax.block_until_ready((Ts_world_root, joints, final_scale))

        # Store the optimized scale
        current_scale = float(final_scale)
        print(f"Optimized scale: {current_scale:.4f}")

        # Convert to lists for playback
        list_T_world_root = [
            jaxlie.SE3(Ts_world_root.wxyz_xyz[t]) for t in range(num_timesteps)
        ]
        list_joints = [joints[t] for t in range(num_timesteps)]

        print("Trajectory optimization complete!")

        with server.atomic():
            # All results are ready; we should be able to do playback.
            playing.disabled = False
            timestep_slider.disabled = False
            # We should also be able to tune weights now.
            for handle in weights._weight_handles.values():
                handle.disabled = False
            rerun_button.disabled = False
            # Update scale display
            scale_label.value = f"{current_scale:.4f}"

            # Update object mesh and collision box visualization with current scale
            scaled_mesh = load_object_mesh(
                asset_dir / "multi_boxes.obj", scale_factor=current_scale
            )
            server.scene.add_mesh_trimesh("/object", scaled_mesh)
            # Update collision box visualization
            scaled_coll_box = pk.collision.Box(
                pose=jaxlie.SE3.from_rotation_and_translation(
                    rotation=coll_box.pose.rotation(),
                    translation=coll_box.pose.translation() * current_scale,
                ),
                size=coll_box.size * current_scale,
            )
            server.scene.add_mesh_trimesh("/box_coll_pk", scaled_coll_box.to_trimesh())

    retarget_trajectory()
    rerun_button.on_click(lambda _: retarget_trajectory())

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            base_frame.wxyz = onp.array(list_T_world_root[tstep].wxyz_xyz[:4])
            base_frame.position = onp.array(list_T_world_root[tstep].wxyz_xyz[4:])
            urdf_vis.update_cfg(onp.array(list_joints[tstep]))

            # Show target keypoints (scaled by optimized scale for visualization)
            keypoints_to_show = (
                motion_unscaled[tstep, onp.array(mocap_indices)] * current_scale
            )
            server.scene.add_point_cloud(
                "/target_keypoints",
                onp.array(keypoints_to_show),
                onp.array((0, 0, 255))[None].repeat(len(mocap_indices), axis=0),
                point_size=0.02,
            )

            # Show object sample points (scaled by optimized scale)
            server.scene.add_point_cloud(
                "/object_points",
                onp.array(object_points_unscaled) * current_scale,
                onp.array((255, 0, 0))[None].repeat(
                    len(object_points_unscaled), axis=0
                ),
                point_size=0.015,
            )

        time.sleep(1 / 30)


# =============================================================================
# Shared Cost Factories
# =============================================================================


def make_laplacian_cost(
    robot: pk.Robot,
    g1_link_indices: jnp.ndarray,
    local_offsets: jnp.ndarray,
    object_points_unscaled: jnp.ndarray,
    adj_matrix: jnp.ndarray,
):
    """Factory for Laplacian mesh deformation cost with optimizable scale.

    Creates a cost that penalizes deviation from target Laplacian coordinates,
    which preserves the local structure of the interaction mesh. The scale factor
    is applied to both object points and target Laplacian (using linearity).

    Args:
        robot: PyRoki robot model.
        g1_link_indices: Robot link indices for keypoints.
        local_offsets: (N, 3) local-frame offsets for each keypoint.
        object_points_unscaled: (num_obj_points, 3) unscaled object surface points.
        adj_matrix: (n_total, n_total) precomputed adjacency matrix.

    Returns:
        A jaxls.Cost.factory-decorated function.
    """

    @jaxls.Cost.factory
    def laplacian_cost(
        var_values: jaxls.VarValues,
        var_T: jaxls.SE3Var,
        var_cfg: jaxls.Var[jnp.ndarray],
        var_scale: ScaleVar,
        target_laplacian_unscaled: jnp.ndarray,
        weight: float,
    ) -> jax.Array:
        robot_cfg = var_values[var_cfg]
        T_world_root = var_values[var_T]
        scale = var_values[var_scale]

        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link

        T_keypoint_links = jaxlie.SE3(T_world_link.wxyz_xyz[..., g1_link_indices, :])
        rotated_offsets = T_keypoint_links.rotation() @ local_offsets
        robot_keypoints = T_keypoint_links.translation() + rotated_offsets

        # Scale object points by the optimizable scale factor
        scaled_object_points = object_points_unscaled * scale
        vertices = jnp.concatenate([robot_keypoints, scaled_object_points], axis=-2)
        current_lap = calculate_laplacian_coordinates_vectorized(vertices, adj_matrix)

        # Scale target Laplacian (Laplacian coordinates scale linearly)
        scaled_target_lap = target_laplacian_unscaled * scale

        return ((current_lap - scaled_target_lap) * weight).flatten()

    return laplacian_cost


def make_world_collision_constraint(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_geom: pk.collision.CollGeom,
):
    """Factory for world collision constraint (hard constraint for IK).

    Args:
        robot: PyRoki robot model.
        robot_coll: Robot collision model.
        world_geom: World collision geometry (unscaled - will be scaled by var_scale).

    Returns:
        A jaxls.Cost.factory-decorated function with constraint_geq_zero.
    """

    @jaxls.Cost.factory(kind="constraint_geq_zero")
    def world_collision_constraint(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jax.Array],
        var_T: jaxls.SE3Var,
        var_scale: ScaleVar,
    ) -> jax.Array:
        cfg = vals[joint_var]
        T_world_root = vals[var_T]
        scale = vals[var_scale]

        # Scale collision geometry by var_scale (both size and position)
        scaled_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=world_geom.pose.rotation(),
            translation=world_geom.pose.translation() * scale,
        )
        scaled_geom = world_geom.__class__(
            pose=scaled_pose,
            size=world_geom.size * scale,
        )

        world_geom_in_root = scaled_geom.transform(T_world_root.inverse())
        dist = robot_coll.compute_world_collision_distance(
            robot, cfg, world_geom_in_root
        )
        margin = 0.02
        return dist.flatten() - margin

    return world_collision_constraint


def make_world_collision_penalty(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_geom: pk.collision.CollGeom,
    margin: float = 0.02,
    penalty_weight: float = 50.0,
):
    """Factory for world collision penalty (soft cost for trajectory optimization).

    Args:
        robot: PyRoki robot model.
        robot_coll: Robot collision model.
        world_geom: World collision geometry (unscaled - will be scaled by var_scale).
        margin: Distance margin for penalty activation.
        penalty_weight: Weight for the penalty term.

    Returns:
        A jaxls.Cost.factory-decorated function.
    """

    @jaxls.Cost.factory
    def world_collision_penalty(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jax.Array],
        var_Ts_world_root: jaxls.SE3Var,
        var_scale: ScaleVar,
    ) -> jax.Array:
        cfg = vals[joint_var]
        Ts_world_root = vals[var_Ts_world_root]
        scale = vals[var_scale]

        # Scale collision geometry by var_scale (both size and position)
        scaled_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=world_geom.pose.rotation(),
            translation=world_geom.pose.translation() * scale,
        )
        scaled_geom = world_geom.__class__(
            pose=scaled_pose,
            size=world_geom.size * scale,
        )

        world_geom_in_root = scaled_geom.transform(Ts_world_root.inverse())
        dist = robot_coll.compute_world_collision_distance(
            robot, cfg, world_geom_in_root
        )

        penalty = jnp.maximum(margin - dist, 0.0) * penalty_weight
        return penalty.flatten()

    return world_collision_penalty


def make_scale_regularization():
    """Factory for scale regularization cost.

    Penalizes deviation of scale from initial value to prevent drift.

    Returns:
        A jaxls.Cost.factory-decorated function.
    """

    @jaxls.Cost.factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_scale: ScaleVar,
        init_scale: float,
        weight: float,
    ) -> jax.Array:
        scale = var_values[var_scale]
        return jnp.array([(scale - init_scale) * weight])

    return scale_regularization


@jdc.jit
def solve_single_frame_ik(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll_list: list[pk.collision.CollGeom],
    target_laplacian_unscaled: jnp.ndarray,
    object_points_unscaled: jnp.ndarray,
    g1_link_indices: jnp.ndarray,
    local_offsets: jnp.ndarray,
    adj_matrix: jnp.ndarray,
    weights: ClimbingRetargetingWeights,
    init_translation: jnp.ndarray,
    prev_T_world_root: jaxlie.SE3 | None,
    prev_joints: jnp.ndarray | None,
    prev_scale: jnp.ndarray | None,
) -> tuple[jaxlie.SE3, jnp.ndarray, jnp.ndarray]:
    """Solve single-frame IK for warm-start initialization with scale optimization.

    Args:
        robot: PyRoki robot model.
        robot_coll: Robot collision model.
        world_coll_list: List of world collision geometries.
        target_laplacian_unscaled: (n_total, 3) unscaled target Laplacian for this frame.
        object_points_unscaled: (num_obj_points, 3) unscaled object surface points.
        g1_link_indices: Robot link indices for keypoints.
        local_offsets: (N, 3) local-frame offsets for each keypoint.
        adj_matrix: (n_total, n_total) precomputed adjacency matrix.
        weights: Retargeting weights.
        init_translation: (3,) initial root translation from mocap centroid.
        prev_T_world_root: Previous frame's root transform (for warm-start).
        prev_joints: Previous frame's joint config (for warm-start).
        prev_scale: Previous frame's scale (for warm-start).

    Returns:
        Tuple of (T_world_root, joints, scale) for this frame.
    """
    var_joints = robot.joint_var_cls(0)
    var_T_world_root = jaxls.SE3Var(0)
    var_scale = ScaleVar(0)

    # Create cost factories
    laplacian_cost = make_laplacian_cost(
        robot, g1_link_indices, local_offsets, object_points_unscaled, adj_matrix
    )
    scale_regularization = make_scale_regularization()

    # Smoothness to previous frame (if available)
    @jaxls.Cost.factory
    def smoothness_to_prev_joints(
        var_values: jaxls.VarValues,
        var_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        if prev_joints is None:
            return jnp.zeros(1)
        return (
            (var_values[var_cfg] - prev_joints) * weights["joint_smoothness"]
        ).flatten()

    @jaxls.Cost.factory
    def smoothness_to_prev_pose(
        var_values: jaxls.VarValues,
        var_T: jaxls.SE3Var,
    ) -> jax.Array:
        if prev_T_world_root is None:
            return jnp.zeros(1)
        return (
            (var_values[var_T].inverse() @ prev_T_world_root).log()
            * weights["pose_smoothness"]
        ).flatten()

    # Build costs
    costs: list[jaxls.Cost] = [
        laplacian_cost(
            var_T_world_root,
            var_joints,
            var_scale,
            target_laplacian_unscaled,
            weights["laplacian"],
        ),
        scale_regularization(var_scale, weights["scale_init"], weights["scale_reg"]),
        smoothness_to_prev_joints(var_joints),
        smoothness_to_prev_pose(var_T_world_root),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full(var_joints.default_factory().shape, weights["rest"]),
        ),
        pk.costs.limit_constraint(
            jax.tree.map(lambda x: x[None], robot),
            robot.joint_var_cls(jnp.arange(1)),
        ),
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], robot_coll),
            robot.joint_var_cls(jnp.arange(1)),
            margin=0.01,
            weight=weights["self_collision"],
        ),
    ]

    # World collision costs (hard constraint for IK)
    for world_coll in world_coll_list:
        world_coll_cost = make_world_collision_constraint(robot, robot_coll, world_coll)
        costs.append(world_coll_cost(var_joints, var_T_world_root, var_scale))

    # Initialize
    if prev_joints is not None:
        init_joint_cfg = prev_joints
    else:
        init_joint_cfg = robot.joint_var_cls.default_factory()

    if prev_T_world_root is not None:
        init_T = prev_T_world_root
    else:
        init_T = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.identity(),
            translation=init_translation,
        )

    if prev_scale is not None:
        init_scale = prev_scale
    else:
        init_scale = jnp.array(weights["scale_init"])

    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[var_joints, var_T_world_root, var_scale],
        )
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make(
                [
                    var_joints.with_value(init_joint_cfg),
                    var_T_world_root.with_value(init_T),
                    var_scale.with_value(init_scale),
                ]
            ),
        )
    )

    return solution[var_T_world_root], solution[var_joints], solution[var_scale]


def compute_warmstart_initialization(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll_list: list[pk.collision.CollGeom],
    target_laplacians_unscaled: jnp.ndarray,
    object_points_unscaled: jnp.ndarray,
    g1_link_indices: jnp.ndarray,
    local_offsets: jnp.ndarray,
    adj_matrix: jnp.ndarray,
    weights: ClimbingRetargetingWeights,
    motion_unscaled: jnp.ndarray,
    mocap_indices: jnp.ndarray,
    keyframe_interval: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute warm-start initialization by solving IK for key frames with scale.

    Args:
        keyframe_interval: Solve IK every N frames.

    Returns:
        Tuple of (init_Ts_wxyz_xyz, init_joints, init_scale) for all timesteps.
    """
    timesteps = target_laplacians_unscaled.shape[0]

    # Select key frame indices
    keyframe_indices = list(range(0, timesteps, keyframe_interval))
    if keyframe_indices[-1] != timesteps - 1:
        keyframe_indices.append(timesteps - 1)

    print(f"Solving IK for {len(keyframe_indices)} key frames...", flush=True)

    # Compute mocap centroids for initialization (scaled by initial scale)
    keypoint_positions = motion_unscaled[:, mocap_indices]
    centroids = keypoint_positions.mean(axis=1) * weights["scale_init"]

    # Solve key frames sequentially
    keyframe_Ts: list[jaxlie.SE3] = []
    keyframe_joints: list[jnp.ndarray] = []
    keyframe_scales: list[jnp.ndarray] = []
    prev_T: jaxlie.SE3 | None = None
    prev_joints: jnp.ndarray | None = None
    prev_scale: jnp.ndarray | None = None

    for i, idx in enumerate(keyframe_indices):
        T, joints, scale = solve_single_frame_ik(
            robot=robot,
            robot_coll=robot_coll,
            world_coll_list=world_coll_list,
            target_laplacian_unscaled=target_laplacians_unscaled[idx],
            object_points_unscaled=object_points_unscaled,
            g1_link_indices=g1_link_indices,
            local_offsets=local_offsets,
            adj_matrix=adj_matrix,
            weights=weights,
            init_translation=centroids[idx],
            prev_T_world_root=prev_T,
            prev_joints=prev_joints,
            prev_scale=prev_scale,
        )
        keyframe_Ts.append(T)
        keyframe_joints.append(joints)
        keyframe_scales.append(scale)
        prev_T = T
        prev_joints = joints
        prev_scale = scale

        if (i + 1) % 5 == 0 or i == len(keyframe_indices) - 1:
            print(
                f"  Solved {i + 1}/{len(keyframe_indices)} key frames "
                f"(scale={float(scale):.4f})",
                flush=True,
            )

    # Use final keyframe scale as initialization for trajectory optimization
    init_scale = keyframe_scales[-1]
    print(f"Warm-start scale: {float(init_scale):.4f}", flush=True)

    # Interpolate between key frames
    init_joints = onp.zeros((timesteps, robot.joint_var_cls.default_factory().shape[0]))
    init_Ts_wxyz_xyz = onp.zeros((timesteps, 7))

    for i in range(len(keyframe_indices) - 1):
        start_idx = keyframe_indices[i]
        end_idx = keyframe_indices[i + 1]
        num_interp = end_idx - start_idx

        # Interpolate joints (linear)
        start_joints = onp.array(keyframe_joints[i])
        end_joints = onp.array(keyframe_joints[i + 1])
        for j in range(num_interp + 1):
            t = j / num_interp if num_interp > 0 else 0.0
            init_joints[start_idx + j] = (1 - t) * start_joints + t * end_joints

        # Interpolate SE3 (slerp for rotation, lerp for translation)
        start_T = keyframe_Ts[i]
        end_T = keyframe_Ts[i + 1]
        start_trans = onp.array(start_T.translation())
        end_trans = onp.array(end_T.translation())
        start_wxyz = onp.array(start_T.rotation().wxyz)
        end_wxyz = onp.array(end_T.rotation().wxyz)

        for j in range(num_interp + 1):
            t = j / num_interp if num_interp > 0 else 0.0
            # Lerp translation
            trans = (1 - t) * start_trans + t * end_trans
            # Slerp rotation (simple linear interp of quaternion + normalize)
            wxyz = (1 - t) * start_wxyz + t * end_wxyz
            wxyz = wxyz / onp.linalg.norm(wxyz)
            init_Ts_wxyz_xyz[start_idx + j] = onp.concatenate([wxyz, trans])

    print("Warm-start initialization complete!", flush=True)
    return jnp.array(init_Ts_wxyz_xyz), jnp.array(init_joints), init_scale


@jdc.jit
def solve_trajectory(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll_list: list[pk.collision.CollGeom],
    target_laplacians_unscaled: jnp.ndarray,
    object_points_unscaled: jnp.ndarray,
    g1_link_indices: jnp.ndarray,
    local_offsets: jnp.ndarray,
    adj_matrix: jnp.ndarray,
    weights: ClimbingRetargetingWeights,
    init_Ts_wxyz_xyz: jnp.ndarray,
    init_joints: jnp.ndarray,
    init_scale: jnp.ndarray,
) -> tuple[jaxlie.SE3, jnp.ndarray, jnp.ndarray]:
    """Solve trajectory optimization for climbing retargeting with scale optimization.

    All timesteps are optimized simultaneously (not sequentially).
    The scale factor is a single global variable optimized across all timesteps.

    Args:
        robot: PyRoki robot model.
        robot_coll: Robot collision model.
        world_coll_list: List of world collision geometries.
        target_laplacians_unscaled: (T, n_total, 3) unscaled target Laplacian coordinates.
        object_points_unscaled: (num_obj_points, 3) unscaled object surface points.
        g1_link_indices: Corresponding G1 robot link indices.
        local_offsets: (N, 3) local-frame offsets for each keypoint.
        adj_matrix: (n_total, n_total) precomputed adjacency matrix.
        weights: Retargeting weights.
        init_Ts_wxyz_xyz: (T, 7) warm-start root transforms.
        init_joints: (T, num_joints) warm-start joint configurations.
        init_scale: Warm-start scale factor.

    Returns:
        Tuple of (Ts_world_root, joints, scale) for all frames.
        - Ts_world_root: SE3 with batch shape (T,)
        - joints: (T, num_joints) array
        - scale: Optimized scale factor
    """
    timesteps = target_laplacians_unscaled.shape[0]

    # Trajectory variables for all timesteps
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    # Single global scale variable (shared across all timesteps)
    var_scale = ScaleVar(0)

    # Batch robot for trajectory (required by pk.costs functions)
    robot_batched = jax.tree.map(lambda x: x[None], robot)
    robot_coll_batched = jax.tree.map(lambda x: x[None], robot_coll)

    # Create cost factories
    laplacian_cost = make_laplacian_cost(
        robot, g1_link_indices, local_offsets, object_points_unscaled, adj_matrix
    )
    scale_regularization = make_scale_regularization()

    # Cost: Joint smoothness between adjacent timesteps
    @jaxls.Cost.factory
    def smoothness_to_prev_joints(
        var_values: jaxls.VarValues,
        var_curr: jaxls.Var[jnp.ndarray],
        var_prev: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        """Smoothness cost for joint configurations between adjacent timesteps."""
        return (
            (var_values[var_curr] - var_values[var_prev]) * weights["joint_smoothness"]
        ).flatten()

    # Cost: Root transform smoothness (SE3 log distance)
    @jaxls.Cost.factory
    def smoothness_to_prev_pose(
        var_values: jaxls.VarValues,
        var_curr: jaxls.Var[jnp.ndarray],
        var_prev: jaxls.Var[jnp.ndarray],
        var_T_curr: jaxls.SE3Var,
        var_T_prev: jaxls.SE3Var,
    ) -> jax.Array:
        """Smoothness cost for root pose trajectory."""
        T_world_joints = var_values[var_T_curr] @ jaxlie.SE3(
            robot.forward_kinematics(cfg=var_values[var_curr])
        )
        prev_T_world_joints = var_values[var_T_prev] @ jaxlie.SE3(
            robot.forward_kinematics(cfg=var_values[var_prev])
        )
        return (
            (prev_T_world_joints.inverse() @ T_world_joints).log()
            * weights["pose_smoothness"]
        ).flatten()

    # Build cost list
    costs: list[jaxls.Cost] = [
        # Laplacian cost for all timesteps (with scale variable)
        laplacian_cost(
            var_Ts_world_root,
            var_joints,
            var_scale,
            target_laplacians_unscaled,
            weights["laplacian"],
        ),
        # Scale regularization (single cost for the global scale)
        scale_regularization(var_scale, weights["scale_init"], weights["scale_reg"]),
        # Joint smoothness between adjacent timesteps
        smoothness_to_prev_joints(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
        ),
        # Root pose smoothness between adjacent timesteps
        smoothness_to_prev_pose(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        # Rest pose regularization
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full((1,) + var_joints.default_factory().shape, weights["rest"]),
        ),
        # Joint limits constraint
        pk.costs.limit_constraint(robot_batched, var_joints),
        # Self-collision cost
        pk.costs.self_collision_cost(
            robot_batched,
            robot_coll_batched,
            var_joints,
            margin=0.01,
            weight=weights["self_collision"],
        ),
    ]

    # World collision costs (soft penalty for trajectory optimization)
    for world_coll in world_coll_list:
        world_coll_cost = make_world_collision_constraint(robot, robot_coll, world_coll)
        costs.append(world_coll_cost(var_joints, var_Ts_world_root, var_scale))

    # Use provided warm-start initialization
    init_Ts = jaxlie.SE3(init_Ts_wxyz_xyz)

    # Solve
    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[var_joints, var_Ts_world_root, var_scale],
        )
        .analyze()
        .solve(
            verbose=True,
            initial_vals=jaxls.VarValues.make(
                [
                    var_joints.with_value(init_joints),
                    var_Ts_world_root.with_value(init_Ts),
                    var_scale.with_value(init_scale),
                ]
            ),
        )
    )

    return solution[var_Ts_world_root], solution[var_joints], solution[var_scale]


if __name__ == "__main__":
    main()
