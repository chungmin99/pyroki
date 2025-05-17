from functools import partial

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

from .._robot import Robot


def _get_actuated_joints_applied_to_target(
    robot: Robot,
    target_joint_idx: jax.Array,
) -> jax.Array:
    """For each joint `i` in the robot, we return an index that is:

    1) -1 if the joint is not in the path to the target link.
    2) an actuated joint index if the joint is in the path to the target link.
        - If the joint `i` is actuated, this is just `i`.
        - If the joint `i` mimics joint `j`, this is set to `j`.

    The inputs and outputs should both be integer arrays of shape
    `(num_joints,)`.
    """

    assert target_joint_idx.shape == ()

    def body_fun(joint_idx, indices):
        # Find the corresponding active actuated joint.
        active_act_joint = jnp.where(
            robot.joints.actuated_indices[joint_idx] != -1,
            # The current joint is actuated.
            robot.joints.actuated_indices[joint_idx],
            # The current joint is not actuated; this is -1 if not a mimic joint.
            robot.joints.mimic_act_indices[joint_idx],
        )

        # Find the parent of the current joint.
        parent_joint = robot.joints.parent_indices[joint_idx]

        # Continue traversing up the kinematic tree, using the parent joint.
        # This value may either go up or down, since there's no guarantee that
        # the kinematic tree is topologically sorted. :-)
        next_indices = indices.at[joint_idx].set(active_act_joint)
        return (parent_joint, next_indices)

    idx_applied_to_target = jnp.full(
        (robot.joints.num_joints,),
        fill_value=-1,
        dtype=jnp.int32,
    )
    idx_applied_to_target = jax.lax.while_loop(
        lambda carry: jnp.any(carry[0] >= 0),
        lambda carry: body_fun(*carry),
        (target_joint_idx, idx_applied_to_target),
    )[-1]
    return idx_applied_to_target


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

    # Get joint twists; these are scaled for mimic joints.
    joint_twists = robot.joints.twists * robot.joints.mimic_multiplier[..., None]

    # Get angular velocity components (omega).
    omega_local = joint_twists[:, 3:]
    omega_wrt_world = Ts_world_joint.rotation() @ omega_local
    omega_wrt_ee = R_ee_world @ omega_wrt_world

    # Get linear velocity components (v).
    vel_local = joint_twists[:, :3]
    vel_wrt_world = Ts_world_joint.rotation() @ vel_local

    # Compute the linear velocity component (v = ω × r + v_joint).
    vel_wrt_world = (
        jnp.cross(
            omega_wrt_world,
            T_world_ee.translation() - Ts_world_joint.translation(),
        )
        + vel_wrt_world
    )
    vel_wrt_ee = R_ee_world @ vel_wrt_world

    # Combine into spatial Jacobian.
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

    # Jacobian of all joints => Jacobian of actuated joints.
    #
    # Because of mimic joints, the Jacobian terms from multiple joints can be
    # applied to a single actuated joint. This is summed!
    jac = (
        jnp.zeros((6, robot.joints.num_actuated_joints))
        .at[:, joints_applied_to_target]
        .add((joints_applied_to_target[None, :] != -1) * jac)
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

    T_world_ee = jaxlie.SE3(Ts_world_link[target_link_index, :])
    pose_error = target_pose.inverse() @ T_world_ee
    return (
        pose_error.log() * jnp.array([pos_weight] * 3 + [ori_weight] * 3),
        # Second argument is cache parameter, which is passed to the custom Jacobian function.
        (Ts_world_joint, Ts_world_link, pose_error),
    )
