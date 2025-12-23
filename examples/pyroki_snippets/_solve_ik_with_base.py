import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls

import numpy as onp

import pyroki as pk


def solve_ik_with_base(
    robot: pk.Robot,
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    fix_base_position: tuple[bool, bool, bool],
    fix_base_orientation: tuple[bool, bool, bool],
    prev_pos: onp.ndarray,
    prev_wxyz: onp.ndarray,
    prev_cfg: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    Solves the basic IK problem for a robot with a mobile base.

    Args:
        robot: PyRoKi Robot.
        target_link_name: str.
        position: onp.ndarray. Shape: (3,).
        wxyz: onp.ndarray. Shape: (4,).
        fix_base_position: Whether to fix the base position (x, y, z).
        fix_base_orientation: Whether to fix the base orientation (w_x, w_y, w_z).
        prev_pos, prev_wxyz, prev_cfg: Previous base position, orientation, and joint configuration, for smooth motion.

    Returns:
        base_pos: onp.ndarray. Shape: (3,).
        base_wxyz: onp.ndarray. Shape: (4,).
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    assert prev_pos.shape == (3,) and prev_wxyz.shape == (4,)
    assert prev_cfg.shape == (robot.joints.num_actuated_joints,)
    target_link_idx = robot.links.names.index(target_link_name)

    T_world_targets = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    base_pose, cfg = _solve_ik_jax(
        robot,
        T_world_targets,
        jnp.array(target_link_idx),
        jnp.array(fix_base_position + fix_base_orientation),
        jnp.array(prev_pos),
        jnp.array(prev_wxyz),
        jnp.array(prev_cfg),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    base_pos = base_pose.translation()
    base_wxyz = base_pose.rotation().wxyz
    assert base_pos.shape == (3,) and base_wxyz.shape == (4,)

    return onp.array(base_pos), onp.array(base_wxyz), onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    T_world_target: jaxlie.SE3,
    target_joint_indices: jnp.ndarray,
    fix_base: jnp.ndarray,
    prev_pos: jnp.ndarray,
    prev_wxyz: jnp.ndarray,
    prev_cfg: jnp.ndarray,
) -> tuple[jaxlie.SE3, jax.Array]:
    joint_var = robot.joint_var_cls(0)

    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
        delta = delta * (1 - fix_base)
        return jaxls.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(prev_wxyz),
            prev_pos,
        ),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...

    base_var = ConstrainedSE3Var(0)

    variables = [joint_var, base_var]
    costs = [
        pk.costs.pose_cost_with_base(
            robot,
            joint_var,
            base_var,
            T_world_target,
            target_joint_indices,
            pos_weight=jnp.array(5.0),
            ori_weight=jnp.array(1.0),
        ),
        pk.costs.rest_with_base_cost(
            joint_var,
            base_var,
            jnp.array(joint_var.default_factory()),
            jnp.array(
                [0.01] * robot.joints.num_actuated_joints
                + [0.1] * 3  # Base position DoF.
                + [0.001] * 3,  # Base orientation DoF.
            ),
        ),
    ]
    costs.append(
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
    )
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [joint_var.with_value(prev_cfg), base_var]
            ),
            verbose=False,
            augmented_lagrangian=jaxls.AugmentedLagrangianConfig(max_iterations=2),
        )
    )
    return sol[base_var], sol[joint_var]
