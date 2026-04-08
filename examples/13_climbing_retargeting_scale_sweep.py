"""Scale Factor Sweep for Climbing Retargeting

Run trajectory optimization with different scale factors to find optimal value.
Based on 13_climbing_retargeting.py but runs headless and reports metrics.

Usage:
    python examples/13_climbing_retargeting_scale_sweep.py --sweep
    python examples/13_climbing_retargeting_scale_sweep.py --scale 0.75
"""

import argparse
import json
import time
from pathlib import Path
from typing import TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import trimesh
from robot_descriptions.loaders.yourdfpy import load_robot_description

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
    joint_smoothness: float
    pose_smoothness: float
    rest: float
    self_collision: float


class SweepResult(TypedDict):
    scale: float
    total_cost: float
    laplacian_cost: float
    smoothness_cost: float
    solve_time: float


# Default weights (same as main example)
DEFAULT_WEIGHTS: ClimbingRetargetingWeights = {
    "laplacian": 3.0,
    "joint_smoothness": 1.0,
    "pose_smoothness": 1.0,
    "rest": 0.1,
    "self_collision": 0.5,
}


def make_laplacian_cost(
    robot: pk.Robot,
    g1_link_indices: jnp.ndarray,
    local_offsets: jnp.ndarray,
    object_points: jnp.ndarray,
    adj_matrix: jnp.ndarray,
):
    """Factory for Laplacian mesh deformation cost."""

    @jaxls.Cost.factory
    def laplacian_cost(
        var_values: jaxls.VarValues,
        var_T: jaxls.SE3Var,
        var_cfg: jaxls.Var[jnp.ndarray],
        target_laplacian: jnp.ndarray,
        weight: float,
    ) -> jax.Array:
        robot_cfg = var_values[var_cfg]
        T_world_root = var_values[var_T]

        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link

        T_keypoint_links = jaxlie.SE3(T_world_link.wxyz_xyz[..., g1_link_indices, :])
        rotated_offsets = T_keypoint_links.rotation() @ local_offsets
        robot_keypoints = T_keypoint_links.translation() + rotated_offsets

        vertices = jnp.concatenate([robot_keypoints, object_points], axis=-2)
        current_lap = calculate_laplacian_coordinates_vectorized(vertices, adj_matrix)

        return ((current_lap - target_laplacian) * weight).flatten()

    return laplacian_cost


def make_world_collision_penalty(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_geom: pk.collision.CollGeom,
    margin: float = 0.02,
    penalty_weight: float = 50.0,
):
    """Factory for world collision penalty."""

    @jaxls.Cost.factory
    def world_collision_penalty(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jax.Array],
        var_Ts_world_root: jaxls.SE3Var,
    ) -> jax.Array:
        cfg = vals[joint_var]
        Ts_world_root = vals[var_Ts_world_root]

        world_geom_in_root = world_geom.transform(Ts_world_root.inverse())
        dist = robot_coll.compute_world_collision_distance(
            robot, cfg, world_geom_in_root
        )

        penalty = jnp.maximum(margin - dist, 0.0) * penalty_weight
        return penalty.flatten()

    return world_collision_penalty


def run_with_scale(scale_factor: float, weights: ClimbingRetargetingWeights) -> SweepResult:
    """Run retargeting with a fixed scale factor, return metrics."""

    # Load robot
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    sphere_json_path = Path(__file__).parent / "assets" / "g1_spheres.json"
    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)
    robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )

    # Load motion data and object with specified scale
    asset_dir = Path(__file__).parent / "retarget_helpers" / "omniretarget_climb_data"
    motion = load_climb_motion(
        asset_dir / "mocap_climb_seq_0_joint_positions_f900-3700.npy",
        scale_factor=scale_factor,
    )
    object_points = load_object_points(
        asset_dir / "multi_boxes.obj",
        sample_count=30,
        scale_factor=scale_factor,
    )
    object_mesh = load_object_mesh(asset_dir / "multi_boxes.obj", scale_factor=scale_factor)

    # Subsample trajectory
    subsample_factor = 1
    motion = motion[::subsample_factor]
    num_timesteps = motion.shape[0]

    # Get retarget indices
    mocap_indices, g1_link_indices, local_offsets = get_climb_retarget_indices(robot)

    # Split up the object mesh into boxes for collision
    meshes = object_mesh.split()
    assert len(meshes) == 3
    box_transforms = []
    box_extents = []
    for mesh in meshes:
        transform, extents = trimesh.bounds.oriented_bounds(mesh)
        box_transforms.append(jaxlie.SE3.from_matrix(transform).inverse())
        box_extents.append(extents)

    coll_box = pk.collision.Box.from_extent(
        extent=onp.array(box_extents),
        position=onp.array([t.translation() for t in box_transforms]),
        wxyz=onp.array([t.rotation().wxyz for t in box_transforms]),
    )

    # Build interaction mesh
    n_robot_pts = len(mocap_indices)
    n_obj_pts = object_points.shape[0]
    n_total_pts = n_robot_pts + n_obj_pts

    demo_robot_pts = motion[0, mocap_indices]
    all_vertices_np = onp.concatenate(
        [onp.array(demo_robot_pts), onp.array(object_points)], axis=0
    )
    _, tetrahedra = create_interaction_mesh(all_vertices_np)
    adj_list = get_adjacency_list(tetrahedra, n_total_pts)
    adj_matrix = adjacency_list_to_matrix(adj_list, n_total_pts)

    # Precompute target Laplacian coordinates
    def compute_target_lap(keypoints_frame):
        robot_pts = keypoints_frame[mocap_indices]
        all_pts = jnp.concatenate([robot_pts, object_points], axis=0)
        return calculate_laplacian_coordinates_vectorized(all_pts, adj_matrix)

    target_laplacians = jax.vmap(compute_target_lap)(motion)

    # Compute mocap centroids for initialization
    keypoint_positions = motion[:, mocap_indices]
    centroids = keypoint_positions.mean(axis=1)

    # Solve trajectory optimization
    start_time = time.time()

    Ts_world_root, joints, cost_info = solve_trajectory_with_costs(
        robot=robot,
        robot_coll=robot_coll,
        world_coll_list=[coll_box],
        target_laplacians=target_laplacians,
        object_points=object_points,
        g1_link_indices=g1_link_indices,
        local_offsets=local_offsets,
        adj_matrix=adj_matrix,
        weights=weights,
        centroids=centroids,
        mocap_indices=mocap_indices,
        motion=motion,
    )
    jax.block_until_ready((Ts_world_root, joints))

    solve_time = time.time() - start_time

    return {
        "scale": scale_factor,
        "total_cost": float(cost_info["total_cost"]),
        "laplacian_cost": float(cost_info["laplacian_cost"]),
        "smoothness_cost": float(cost_info["smoothness_cost"]),
        "solve_time": solve_time,
    }


@jdc.jit
def solve_trajectory_with_costs(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll_list: list[pk.collision.CollGeom],
    target_laplacians: jnp.ndarray,
    object_points: jnp.ndarray,
    g1_link_indices: jnp.ndarray,
    local_offsets: jnp.ndarray,
    adj_matrix: jnp.ndarray,
    weights: ClimbingRetargetingWeights,
    centroids: jnp.ndarray,
    mocap_indices: jnp.ndarray,
    motion: jnp.ndarray,
) -> tuple[jaxlie.SE3, jnp.ndarray, dict]:
    """Solve trajectory optimization and return solution with cost metrics."""
    timesteps = target_laplacians.shape[0]

    # Variables
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))

    robot_batched = jax.tree.map(lambda x: x[None], robot)
    robot_coll_batched = jax.tree.map(lambda x: x[None], robot_coll)

    # Create costs
    laplacian_cost = make_laplacian_cost(
        robot, g1_link_indices, local_offsets, object_points, adj_matrix
    )

    @jaxls.Cost.factory
    def smoothness_to_prev_joints(
        var_values: jaxls.VarValues,
        var_curr: jaxls.Var[jnp.ndarray],
        var_prev: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        return (
            (var_values[var_curr] - var_values[var_prev]) * weights["joint_smoothness"]
        ).flatten()

    @jaxls.Cost.factory
    def smoothness_to_prev_pose(
        var_values: jaxls.VarValues,
        var_curr: jaxls.Var[jnp.ndarray],
        var_prev: jaxls.Var[jnp.ndarray],
        var_T_curr: jaxls.SE3Var,
        var_T_prev: jaxls.SE3Var,
    ) -> jax.Array:
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

    # Build costs
    costs: list[jaxls.Cost] = [
        laplacian_cost(
            var_Ts_world_root, var_joints, target_laplacians, weights["laplacian"]
        ),
        smoothness_to_prev_joints(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
        ),
        smoothness_to_prev_pose(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full((1,) + var_joints.default_factory().shape, weights["rest"]),
        ),
        pk.costs.limit_constraint(robot_batched, var_joints),
        pk.costs.self_collision_cost(
            robot_batched,
            robot_coll_batched,
            var_joints,
            margin=0.01,
            weight=weights["self_collision"],
        ),
    ]

    for world_coll in world_coll_list:
        world_coll_cost = make_world_collision_penalty(robot, robot_coll, world_coll)
        costs.append(world_coll_cost(var_joints, var_Ts_world_root))

    # Simple initialization from centroids
    init_joints = jnp.tile(
        robot.joint_var_cls.default_factory()[None], (timesteps, 1)
    )
    init_Ts_wxyz_xyz = jnp.concatenate(
        [
            jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0])[None], (timesteps, 1)),
            centroids,
        ],
        axis=-1,
    )
    init_Ts = jaxlie.SE3(init_Ts_wxyz_xyz)

    # Solve
    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[var_joints, var_Ts_world_root],
        )
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make(
                [
                    var_joints.with_value(init_joints),
                    var_Ts_world_root.with_value(init_Ts),
                ]
            ),
        )
    )

    # Compute cost metrics on solution
    sol_joints = solution[var_joints]
    sol_Ts = solution[var_Ts_world_root]

    # Compute Laplacian cost by vmapping over timesteps
    def compute_lap_for_timestep(cfg, T_world_root_wxyz_xyz):
        T_world_root = jaxlie.SE3(T_world_root_wxyz_xyz)
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg))
        T_world_link = T_world_root @ T_root_link
        T_keypoint_links = jaxlie.SE3(T_world_link.wxyz_xyz[g1_link_indices, :])
        rotated_offsets = T_keypoint_links.rotation() @ local_offsets
        robot_keypoints = T_keypoint_links.translation() + rotated_offsets
        vertices = jnp.concatenate([robot_keypoints, object_points], axis=0)
        return calculate_laplacian_coordinates_vectorized(vertices, adj_matrix)

    current_lap = jax.vmap(compute_lap_for_timestep)(sol_joints, sol_Ts.wxyz_xyz)
    lap_residual = (current_lap - target_laplacians) * weights["laplacian"]
    laplacian_cost_val = jnp.sum(lap_residual**2)

    # Compute smoothness cost (joints)
    joint_diff = sol_joints[1:] - sol_joints[:-1]
    smoothness_cost_val = jnp.sum((joint_diff * weights["joint_smoothness"]) ** 2)

    total_cost = laplacian_cost_val + smoothness_cost_val

    cost_info = {
        "total_cost": total_cost,
        "laplacian_cost": laplacian_cost_val,
        "smoothness_cost": smoothness_cost_val,
    }

    return sol_Ts, sol_joints, cost_info


def main():
    parser = argparse.ArgumentParser(description="Scale factor sweep for climbing retargeting")
    parser.add_argument("--scale", type=float, default=0.742, help="Single scale factor to test")
    parser.add_argument("--sweep", action="store_true", help="Run full sweep over scale factors")
    args = parser.parse_args()

    if args.sweep:
        scales = [0.68, 0.70, 0.72, 0.742, 0.76, 0.78, 0.80]
        results: list[SweepResult] = []

        print("Running scale factor sweep...")
        print("=" * 70)

        for scale in scales:
            print(f"\nTesting scale = {scale:.3f}...")
            result = run_with_scale(scale, DEFAULT_WEIGHTS)
            results.append(result)
            print(f"  Total cost: {result['total_cost']:.4f}")
            print(f"  Laplacian cost: {result['laplacian_cost']:.4f}")
            print(f"  Solve time: {result['solve_time']:.2f}s")

        # Print summary table
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Scale':<8} | {'Laplacian Cost':<16} | {'Total Cost':<14} | {'Time (s)':<10}")
        print("-" * 70)

        best_result = min(results, key=lambda r: r["laplacian_cost"])
        for r in results:
            marker = " *" if r["scale"] == best_result["scale"] else ""
            print(
                f"{r['scale']:<8.3f} | {r['laplacian_cost']:<16.4f} | "
                f"{r['total_cost']:<14.4f} | {r['solve_time']:<10.2f}{marker}"
            )

        print("-" * 70)
        print(f"* Best scale: {best_result['scale']:.3f} (lowest Laplacian cost)")
        print(f"  Current default: 0.742")

        if best_result["scale"] != 0.742:
            baseline = next(r for r in results if r["scale"] == 0.742)
            improvement = (baseline["laplacian_cost"] - best_result["laplacian_cost"]) / baseline["laplacian_cost"] * 100
            print(f"  Improvement over default: {improvement:.1f}%")
    else:
        print(f"Testing scale = {args.scale}...")
        result = run_with_scale(args.scale, DEFAULT_WEIGHTS)
        print(f"\nResults for scale = {result['scale']:.3f}:")
        print(f"  Total cost: {result['total_cost']:.4f}")
        print(f"  Laplacian cost: {result['laplacian_cost']:.4f}")
        print(f"  Smoothness cost: {result['smoothness_cost']:.4f}")
        print(f"  Solve time: {result['solve_time']:.2f}s")


if __name__ == "__main__":
    main()
