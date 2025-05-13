from functools import partial

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

from .._robot import Robot


def _get_actuated_joints_applied_to_target(
    robot: Robot,
    target_joint_idx: jax.Array,
):
    """Get the indices of the actuated joints that affect the pose of the target link.

    We return an array with shape (num_joints,), where each element is either
    (a) an actuated joint index or (b) -1 if the joint is not in the path to
    the target link.
    """

    def scan_fn(current_parent_idx: jax.Array, i: jax.Array):
        # Check if this joint is in the path to the target
        is_in_path = i == current_parent_idx
        act_joint_idx = jnp.where(is_in_path, robot.joints.actuated_indices[..., i], -1)
        current_parent_idx = jnp.where(
            is_in_path, robot.joints.parent_indices[..., i], current_parent_idx
        )
        return current_parent_idx, act_joint_idx

    return jax.lax.scan(
        scan_fn,
        init=target_joint_idx,
        xs=jnp.arange(robot.joints.num_joints),
        reverse=True,
    )[1]


_PoseCostJacCache = tuple[jax.Array, jax.Array, jaxlie.SE3]


def pose_cost_analytic_jac(
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
) -> jaxls.Cost:
    # We only check shape lengths because there might be (1,) axes for
    # broadcasting reasons.
    assert (
        len(target_link_index.shape)
        == len(jnp.asarray(joint_var.id).shape)
        == len(robot.joints.twists.shape[:-2])
    ), "Batch axes of inputs should match"

    # Broadcast the inputs for _get_actuated_joints_applied_to_target().
    # Excluding the weights for now...
    batch_axes = jnp.broadcast_shapes(
        target_pose.get_batch_axes(),
        jnp.asarray(joint_var.id).shape,
        target_pose.get_batch_axes(),
        target_link_index.shape,
    )
    broadcast_batch_axes = partial(
        jax.tree.map,
        lambda x: jnp.broadcast_to(x, batch_axes + x.shape[len(batch_axes) :]),
    )
    get_actuated_joints = _get_actuated_joints_applied_to_target
    for _ in range(len(batch_axes)):
        get_actuated_joints = jax.vmap(get_actuated_joints)

    # Compute applied joints.
    robot = broadcast_batch_axes(robot)
    base_link_mask = robot.links.parent_joint_indices == -1
    parent_joint_indices = jnp.where(
        base_link_mask, 0, robot.links.parent_joint_indices
    )
    target_joint_idx = parent_joint_indices[
        tuple(jnp.arange(d) for d in parent_joint_indices.shape[:-1])
        + (target_link_index,)
    ]
    joints_applied_to_target = get_actuated_joints(
        broadcast_batch_axes(robot), broadcast_batch_axes(target_joint_idx)
    )

    return _pose_cost_analytical_jac(
        robot,
        joint_var,
        target_pose,
        target_link_index,
        pos_weight,
        ori_weight,
        joints_applied_to_target,
    )


# It's nice to pass arguments in explicitly instead of via closure in the
# `pose_cost_analytic_jac` wrapper. It helps jaxls vectorize repeated costs.
def _pose_cost_jac(
    vals: jaxls.VarValues,
    jac_cache: _PoseCostJacCache,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    joints_applied_to_target: jax.Array,
) -> jax.Array:
    """Jacobian for pose cost with analytic computation."""
    del vals, joint_var, target_pose  # Unused!
    Ts_world_joint, Ts_world_link, pose_error = jac_cache

    T_world_ee = jaxlie.SE3(Ts_world_link[target_link_index])
    Ts_world_joint = jaxlie.SE3(Ts_world_joint)

    R_ee_world = T_world_ee.rotation().inverse()

    # Get joint twists
    joint_twists = robot.joints.twists

    # Get angular velocity components (omega)
    omega_local = joint_twists[..., 3:]
    omega_wrt_world = Ts_world_joint.rotation() @ omega_local
    omega_wrt_ee = R_ee_world @ omega_wrt_world

    # Get linear velocity components (v)
    vel_local = joint_twists[..., :3]
    vel_wrt_world = Ts_world_joint.rotation() @ vel_local

    # Compute the linear velocity component (v = ω × r + v_joint)
    vel_wrt_world = (
        jnp.cross(
            omega_wrt_world,
            T_world_ee.translation() - Ts_world_joint.translation(),
        )
        + vel_wrt_world
    )
    vel_wrt_ee = R_ee_world @ vel_wrt_world

    # Combine into spatial jacobian
    jac = jnp.where(
        joints_applied_to_target[:, None] != -1,
        jnp.concatenate(
            [
                vel_wrt_ee,
                omega_wrt_ee,
            ],
            axis=1,
        ),
        0.0,
    ).T
    jac = pose_error.jlog() @ jac

    # Fixed‑size list of kinematic joints that are actually actuated
    kin_idx = jnp.nonzero(robot.joints.actuated_indices != -1,
                          size=robot.joints.actuated_indices.size,
                          fill_value=0)[0] # (num_joints,)

    # Map each kept joint → its actuator id (may repeat for mimic joints)
    act_idx = robot.joints.actuated_indices[kin_idx] # (num_joints,)

    # Scatter‑add merges mimic joints into their single actuator column
    jac = (
        jnp.zeros((6, robot.joints.num_actuated_joints), jac.dtype)
          .at[:, act_idx]
          .add(jac[:, kin_idx])
    )

    # Apply weights
    weights = jnp.array([pos_weight] * 3 + [ori_weight] * 3)
    return jac * weights[:, None]


@jaxls.Cost.create_factory(jac_custom_with_cache_fn=_pose_cost_jac)
def _pose_cost_analytical_jac(
    vals: jaxls.VarValues,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    joints_applied_to_target: jax.Array,
) -> tuple[jax.Array, _PoseCostJacCache]:
    """Computes the residual for matching link poses to target poses."""
    del joints_applied_to_target
    assert target_link_index.dtype == jnp.int32
    joint_cfg = vals[joint_var]

    Ts_world_joint = robot._forward_kinematics_joints(joint_cfg)
    Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)

    T_world_ee = jaxlie.SE3(Ts_world_link[..., target_link_index, :])
    pose_error = target_pose.inverse() @ T_world_ee
    return (
        pose_error.log() * jnp.array([pos_weight] * 3 + [ori_weight] * 3),
        # Second argument is cache parameter, which is passed to the custom Jacobian function.
        (Ts_world_joint, Ts_world_link, pose_error),
    )
