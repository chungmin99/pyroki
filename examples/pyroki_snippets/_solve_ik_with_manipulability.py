"""
Solves the basic IK problem.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls

import numpy as onp

import pyroki as pk


def solve_ik_with_manipulability(
    robot: pk.Robot,
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    manipulability_weight: float = 0.0,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot, with manipulability cost.

    Args:
        robot: PyRoKi Robot.
        target_link_name: str.
        position: onp.ndarray. Shape: (3,).
        wxyz: onp.ndarray. Shape: (4,).
        manipulability_weight: float. Weight for the manipulability cost.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)

    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_idx = robot.links.names.index(target_link_name)

    T_world_target = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    cfg = _solve_ik_jax(
        robot,
        T_world_target,
        jnp.array(target_link_idx),
        jnp.array(manipulability_weight),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    T_world_target: jaxlie.SE3,
    target_joint_idx: jnp.ndarray,
    manipulability_weight: jnp.ndarray,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            T_world_target,
            target_joint_idx,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            jnp.array(joint_var.default_factory()),
            jnp.array([0.01] * robot.joints.num_actuated_joints),
        ),
        pk.costs.manipulability_cost(
            robot,
            joint_var,
            target_joint_idx,
            manipulability_weight,
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
        .solve(verbose=False, linear_solver="dense_cholesky")
    )
    return sol[joint_var]
