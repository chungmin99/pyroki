"""
Solves the basic IK problem with collision avoidance.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: Sequence[str]. Length: num_targets.
        position: ArrayLike. Shape: (num_targets, 3), or (3,).
        wxyz: ArrayLike. Shape: (num_targets, 4), or (4,).

    Returns:
        cfg: ArrayLike. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_idx = robot.links.names.index(target_link_name)

    T_world_targets = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    cfg = _solve_ik_with_collision_jax(
        robot,
        coll,
        world_coll_list,
        T_world_targets,
        jnp.array(target_link_idx),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_ik_with_collision_jax(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    T_world_target: jaxlie.SE3,
    target_link_index: jax.Array,
) -> jax.Array:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]

    # Weights and margins defined directly in factors
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            pos_weight=5.0,
            ori_weight=1.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll=coll,
            joint_var=joint_var,
            margin=0.02,
            weight=5.0,
        ),
    ]
    costs.append(
        pk.costs.limit_constraint(
            robot,
            joint_var,
        )
    )
    costs.extend(
        [
            pk.costs.world_collision_constraint(
                robot, coll, joint_var, world_coll, 0.05
            )
            for world_coll in world_coll_list
        ]
    )

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            augmented_lagrangian=jaxls.AugmentedLagrangianConfig(max_iterations=5),
        )
    )
    return sol[joint_var]
