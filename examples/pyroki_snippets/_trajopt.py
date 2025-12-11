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

    # --- Soft costs ---
    costs: list[jaxls.Cost] = [
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.01])[None],
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([1])[None],
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
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            traj_vars,
            0.02,
            5.0,
        ),
    ]

    # --- Constraints (augmented Lagrangian penalties) ---
    # Joint limits.
    costs.append(pk.costs.limit_constraint(robot, traj_vars))

    # Start / end pose constraints.
    @jaxls.Cost.factory(kind="constraint_eq_zero", name="start_pose_constraint")
    def start_pose_constraint(vals: jaxls.VarValues, var) -> jax.Array:
        return (vals[var] - start_cfg).flatten()

    @jaxls.Cost.factory(kind="constraint_eq_zero", name="end_pose_constraint")
    def end_pose_constraint(vals: jaxls.VarValues, var) -> jax.Array:
        return (vals[var] - end_cfg).flatten()

    costs.append(start_pose_constraint(robot.joint_var_cls(jnp.arange(0, 2))))
    costs.append(
        end_pose_constraint(robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps)))
    )

    # Velocity limits.
    costs.append(
        pk.costs.limit_velocity_constraint(
            robot,
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            dt,
        )
    )

    for world_coll_obj in world_coll:
        costs.append(
            pk.costs.world_collision_constraint(
                robot,
                robot_coll,
                traj_vars,
                jax.tree.map(lambda x: x[None], world_coll_obj),
                0.01,
            )
        )

    # 4. Solve the optimization problem with augmented Lagrangian for constraints.
    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[traj_vars],
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
    variables = [joint_vars]

    # Soft costs: pose matching, regularization, self-collision
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var_0,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_0), target_position_0
            ),
            jnp.array(target_link_index),
            jnp.array([10.0] * 3),
            jnp.array([1.0] * 3),
        ),
        pk.costs.pose_cost(
            robot,
            joint_var_1,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_1), target_position_1
            ),
            jnp.array(target_link_index),
            jnp.array([10.0] * 3),
            jnp.array([1.0] * 3),
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

    # Small cost to encourage the start + end configs to be close to each other.
    @jaxls.Cost.factory(name="JointSimilarityCost")
    def joint_similarity_cost(vals, var_0, var_1):
        return (vals[var_0] - vals[var_1]).flatten()

    costs.append(joint_similarity_cost(joint_var_0, joint_var_1))

    # World collision as soft cost (more robust for IK initialization)
    costs.extend(
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

    # Constraint: joint limits
    costs.append(
        pk.costs.limit_constraint(
            jax.tree.map(lambda x: x[None], robot),
            joint_vars,
        ),
    )

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            verbose=False,
            augmented_lagrangian=jaxls.AugmentedLagrangianConfig(max_iterations=5),
        )
    )
    return sol[joint_var_0], sol[joint_var_1]
