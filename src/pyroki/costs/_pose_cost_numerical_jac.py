import jax
import jax.numpy as jnp
import jaxlie
import jaxls

from .._robot import Robot

_PoseCostJacCache = tuple[jax.Array, jax.Array, jaxlie.SE3]


def _pose_cost_jac(
    vals: jaxls.VarValues,
    jac_cache: _PoseCostJacCache,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    eps: float = 1e-4,
) -> jax.Array:
    """Jacobian for pose cost with numerical computation."""
    joint_cfg = vals[joint_var]
    _, _, pose_error = jac_cache

    def finite_difference_jac(idx: jax.Array) -> jax.Array:
        joint_cfg_perturbed = joint_cfg.at[idx].add(eps)
        T_world_ee = jaxlie.SE3(
            robot.forward_kinematics(joint_cfg_perturbed)[..., target_link_index, :]
        )
        perturbed_pose_error = target_pose.inverse() @ T_world_ee
        err_diff = perturbed_pose_error.log() - pose_error.log()
        return err_diff / eps

    jac = jax.vmap(finite_difference_jac)(jnp.arange(joint_cfg.shape[-1])).T
    assert jac.shape == (6, robot.joints.num_actuated_joints)

    return jac * jnp.array([pos_weight] * 3 + [ori_weight] * 3)[:, None]


@jaxls.Cost.create_factory(jac_custom_with_cache_fn=_pose_cost_jac)
def pose_cost_numerical_jac(
    vals: jaxls.VarValues,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    eps: float = 1e-4,
) -> tuple[jax.Array, _PoseCostJacCache]:
    """Computes the residual for matching link poses to target poses."""
    del eps  # Unused!
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
