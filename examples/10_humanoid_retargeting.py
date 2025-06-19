"""Humanoid Retargeting

Simpler motion retargeting to the G1 humanoid.
"""

import time
from pathlib import Path
from typing import Tuple, TypedDict

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

from retarget_helpers._utils import (
    SMPL_JOINT_NAMES,
    create_conn_tree,
    get_humanoid_retarget_indices,
)


class RetargetingWeights(TypedDict):
    local_alignment: float
    """Local alignment weight, by matching the relative joint/keypoint positions and angles."""
    global_alignment: float
    """Global alignment weight, by matching the keypoint positions to the robot."""


def main():
    """Main function for humanoid retargeting."""

    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    # Load source motion data:
    # - keypoints [N, 45, 3],
    # - left/right foot contact (boolean) 2 x [N],
    # - heightmap [H, W].
    asset_dir = Path(__file__).parent / "retarget_helpers" / "humanoid"
    smpl_keypoints = onp.load(asset_dir / "smpl_keypoints.npy")
    is_left_foot_contact = onp.load(asset_dir / "left_foot_contact.npy")
    is_right_foot_contact = onp.load(asset_dir / "right_foot_contact.npy")
    heightmap = onp.load(asset_dir / "heightmap.npy")

    num_timesteps = smpl_keypoints.shape[0]
    assert smpl_keypoints.shape == (num_timesteps, 45, 3)
    assert is_left_foot_contact.shape == (num_timesteps,)
    assert is_right_foot_contact.shape == (num_timesteps,)

    heightmap = pk.collision.Heightmap(
        pose=jaxlie.SE3.identity(),
        size=jnp.array([0.01, 0.01, 1.0]),
        height_data=heightmap,
    )

    # Get the left and right foot keypoints, projected on the heightmap.
    left_foot_keypoint_idx = SMPL_JOINT_NAMES.index("left_foot")
    right_foot_keypoint_idx = SMPL_JOINT_NAMES.index("right_foot")
    left_foot_keypoints = smpl_keypoints[..., left_foot_keypoint_idx, :].reshape(-1, 3)
    right_foot_keypoints = smpl_keypoints[..., right_foot_keypoint_idx, :].reshape(
        -1, 3
    )
    left_foot_keypoints = heightmap.project_points(left_foot_keypoints)
    right_foot_keypoints = heightmap.project_points(right_foot_keypoints)

    smpl_joint_retarget_indices, g1_joint_retarget_indices = (
        get_humanoid_retarget_indices()
    )
    smpl_mask = create_conn_tree(robot, g1_joint_retarget_indices)

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())

    weights = pk.viewer.WeightTuner(
        server,
        RetargetingWeights(  # type: ignore
            local_alignment=2.0,
            global_alignment=1.0,
        ),
    )

    Ts_world_root, joints = None, None

    def generate_trajectory():
        nonlocal Ts_world_root, joints
        gen_button.disabled = True
        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            target_keypoints=smpl_keypoints,
            smpl_joint_retarget_indices=smpl_joint_retarget_indices,
            g1_joint_retarget_indices=g1_joint_retarget_indices,
            smpl_mask=smpl_mask,
            weights=weights.get_weights(),  # type: ignore
        )
        gen_button.disabled = False

    gen_button = server.gui.add_button("Retarget!")
    gen_button.on_click(lambda _: generate_trajectory())

    generate_trajectory()
    assert Ts_world_root is not None and joints is not None

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame.position = onp.array(Ts_world_root.wxyz_xyz[tstep][4:])
            urdf_vis.update_cfg(onp.array(joints[tstep]))
            server.scene.add_point_cloud(
                "/target_keypoints",
                onp.array(smpl_keypoints[tstep]),
                onp.array((0, 0, 255))[None].repeat(45, axis=0),
                point_size=0.01,
            )

        time.sleep(0.05)


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    target_keypoints: jnp.ndarray,
    smpl_joint_retarget_indices: jnp.ndarray,
    g1_joint_retarget_indices: jnp.ndarray,
    smpl_mask: jnp.ndarray,
    weights: RetargetingWeights,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the retargeting problem."""

    n_retarget = len(smpl_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    # Robot properties.
    # - Joints that should move less for natural humanoid motion.
    joints_to_move_less = jnp.array(
        [
            robot.joints.actuated_names.index(name)
            for name in ["left_hip_yaw_joint", "right_hip_yaw_joint", "torso_joint"]
        ]
    )

    # Variables.
    class SmplJointsScaleVarG1(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((n_retarget, n_retarget))
    ): ...

    class OffsetVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros((3,))): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_smpl_joints_scale = SmplJointsScaleVarG1(jnp.zeros(timesteps))
    var_offset = OffsetVar(jnp.zeros(timesteps))

    # Costs.
    costs: list[jaxls.Cost] = []

    @jaxls.Cost.create_factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_smpl_joints_scale: SmplJointsScaleVarG1,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Retargeting factor, with a focus on:
        - matching the relative joint/keypoint positions (vectors).
        - and matching the relative angles between the vectors.
        """
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link

        smpl_pos = keypoints[jnp.array(smpl_joint_retarget_indices)]
        robot_pos = T_world_link.translation()[jnp.array(g1_joint_retarget_indices)]

        # NxN grid of relative positions.
        delta_smpl = smpl_pos[:, None] - smpl_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        # Vector regularization.
        position_scale = var_values[var_smpl_joints_scale][..., None]
        residual_position_delta = (
            (delta_smpl - delta_robot * position_scale)
            * (1 - jnp.eye(delta_smpl.shape[0])[..., None])
            * smpl_mask[..., None]
        )

        # Vector angle regularization.
        delta_smpl_normalized = delta_smpl / jnp.linalg.norm(
            delta_smpl + 1e-6, axis=-1, keepdims=True
        )
        delta_robot_normalized = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True
        )
        residual_angle_delta = 1 - (delta_smpl_normalized * delta_robot_normalized).sum(
            axis=-1
        )
        residual_angle_delta = (
            residual_angle_delta
            * (1 - jnp.eye(residual_angle_delta.shape[0]))
            * smpl_mask
        )

        residual = (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()]
            )
            * weights["local_alignment"]
        )
        return residual

    @jaxls.Cost.create_factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Soft cost to align the human keypoints to the robot, in the world frame."""
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[g1_joint_retarget_indices]
        keypoint_pos = keypoints[smpl_joint_retarget_indices]
        return (link_pos - keypoint_pos).flatten() * weights["global_alignment"]

    costs = [
        # Costs that are relatively self-contained to the robot.
        retargeting_cost(
            var_Ts_world_root,
            var_joints,
            var_smpl_joints_scale,
            target_keypoints,
        ),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            100.0,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([0.2]),
        ),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full(var_joints.default_factory().shape, 0.2)
            .at[joints_to_move_less]
            .set(2.0)[None],
        ),
        # Costs that are scene-centric.
        pc_alignment_cost(
            var_Ts_world_root,
            var_joints,
            target_keypoints,
        ),
    ]

    solution = (
        jaxls.LeastSquaresProblem(
            costs, [var_joints, var_Ts_world_root, var_smpl_joints_scale, var_offset]
        )
        .analyze()
        .solve()
    )
    transform = solution[var_Ts_world_root]
    offset = solution[var_offset]
    transform = jaxlie.SE3.from_translation(offset) @ transform
    return transform, solution[var_joints]


if __name__ == "__main__":
    main()
