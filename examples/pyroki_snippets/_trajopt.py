from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax.typing import ArrayLike


def solve_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_position: ArrayLike,
    start_wxyz: ArrayLike,
    end_position: ArrayLike,
    end_wxyz: ArrayLike,
    timesteps: int,
    dt: float,
) -> ArrayLike:
    if isinstance(start_position, onp.ndarray):
        np = onp
    elif isinstance(start_position, jnp.ndarray):
        np = jnp
    else:
        raise ValueError(f"Invalid type for `ArrayLike`: {type(start_position)}")

    # 1. Solve IK for the start and end poses.
    target_link_index = robot.links.names.index(target_link_name)
    start_cfg, end_cfg = solve_iks_with_collision(
        robot=robot,
        coll=robot_coll,
        world_coll_list=world_coll,
        target_link_index=target_link_index,
        target_position_0=jnp.array(start_position),
        target_wxyz_0=jnp.array(start_wxyz),
        target_position_1=jnp.array(end_position),
        target_wxyz_1=jnp.array(end_wxyz),
    )

    # 2. Initialize the trajectory through linearly interpolating the start and end poses.
    init_traj = jnp.linspace(start_cfg, end_cfg, timesteps)

    # 3. Optimize the trajectory.
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))

    robot = jax.tree.map(lambda x: x[None], robot)  # Add batch dimension.
    robot_coll = jax.tree.map(lambda x: x[None], robot_coll)  # Add batch dimension.

    # Basic regularization / limit costs.
    factors: list[jaxls.Cost] = [
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.01])[None],
        ),
        pk.costs.limit_cost(
            robot,
            traj_vars,
            jnp.array([100.0])[None],
        ),
    ]

    # Collision avoidance.
    def compute_world_coll_residual(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        robot_coll: pk.collision.RobotCollision,
        world_coll_obj: pk.collision.CollGeom,
        prev_traj_vars: jaxls.Var[jax.Array],
        curr_traj_vars: jaxls.Var[jax.Array],
    ):
        coll = robot_coll.get_swept_capsules(
            robot, vals[prev_traj_vars], vals[curr_traj_vars]
        )
        dist = pk.collision.collide(
            coll.reshape((-1, 1)), world_coll_obj.reshape((1, -1))
        )
        colldist = pk.collision.colldist_from_sdf(dist, 0.1)
        return (colldist * 20.0).flatten()

    for world_coll_obj in world_coll:
        factors.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot,
                    robot_coll,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                name="World Collision (sweep)",
            )
        )

    # Start / end pose constraints.
    factors.extend(
        [
            jaxls.Cost(
                lambda vals, var: ((vals[var] - start_cfg) * 100.0).flatten(),
                (robot.joint_var_cls(jnp.arange(0, 2)),),
                name="start_pose_constraint",
            ),
            jaxls.Cost(
                lambda vals, var: ((vals[var] - end_cfg) * 100.0).flatten(),
                (robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps)),),
                name="end_pose_constraint",
            ),
        ]
    )

    # Velocity / acceleration / jerk minimization.
    factors.extend(
        [
            pk.costs.smoothness_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                jnp.array([0.1])[None],
            ),
            pk.costs.five_point_velocity_cost(
                robot,
                robot.joint_var_cls(jnp.arange(4, timesteps)),
                robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                dt,
                jnp.array([10.0])[None],
            ),
            pk.costs.five_point_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(4, timesteps)),
                robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                dt,
                jnp.array([0.1])[None],
            ),
            pk.costs.five_point_jerk_cost(
                robot.joint_var_cls(jnp.arange(6, timesteps)),
                robot.joint_var_cls(jnp.arange(5, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(4, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(2, timesteps - 4)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 5)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 6)),
                dt,
                jnp.array([0.1])[None],
            ),
        ]
    )

    # 4. Solve the optimization problem.
    solution = (
        jaxls.LeastSquaresProblem(
            factors,
            [traj_vars],
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
        )
    )
    return np.array(solution[traj_vars])


@jdc.jit
def solve_iks_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    target_position_0: jax.Array,
    target_wxyz_0: jax.Array,
    target_position_1: jax.Array,
    target_wxyz_1: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    joint_var_0 = robot.joint_var_cls(0)
    joint_var_1 = robot.joint_var_cls(1)
    joint_vars = robot.joint_var_cls(jnp.arange(2))
    vars = [joint_vars]

    # Weights and margins defined directly in factors.
    factors = [
        pk.costs.pose_cost(
            robot,
            joint_var_0,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_0), target_position_0
            ),
            jnp.array(target_link_index),
            jnp.array([5.0] * 3),
            jnp.array([1.0] * 3),
        ),
        pk.costs.pose_cost(
            robot,
            joint_var_1,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_1), target_position_1
            ),
            jnp.array(target_link_index),
            jnp.array([5.0] * 3),
            jnp.array([1.0] * 3),
        ),
    ]
    factors.extend(
        [
            pk.costs.limit_cost(
                jax.tree.map(lambda x: x[None], robot),
                joint_vars,
                jnp.array(100.0),
            ),
            pk.costs.rest_cost(
                joint_vars,
                jnp.array(joint_vars.default_factory()[None]),
                jnp.array(0.001),
            ),
            pk.costs.self_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], coll),
                joint_vars,
                0.02,
                5.0,
            ),
        ]
    )
    factors.extend(
        [
            pk.costs.world_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], coll),
                joint_vars,
                jax.tree.map(lambda x: x[None], world_coll),
                0.05,
                10.0,
            )
            for world_coll in world_coll_list
        ]
    )

    # Small cost to encourage the start + end configs to be close to each other.
    @jaxls.Cost.create_factory(name="JointSimilarityCost")
    def joint_similarity_cost(vals, var_0, var_1):
        return ((vals[var_0] - vals[var_1]) * 0.01).flatten()

    factors.append(joint_similarity_cost(joint_var_0, joint_var_1))

    sol = jaxls.LeastSquaresProblem(factors, vars).analyze().solve(verbose=False)
    return sol[joint_var_0], sol[joint_var_1]
