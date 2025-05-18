"""Modified version of cuRobo's IK benchmark script:

    https://github.com/NVlabs/curobo/blob/0a50de1ba72db304195d59d9d0b1ed269696047f/benchmark/ik_benchmark.py

Compares a PyRoki-based IK solver ("IK-Beam") against cuRobo's IK solver.

Example outputs, on RTX 4090:

+----+------------+--------------+-------------------+---------------+------------------+------------------+-------------------+---------------+----------------------+------------------+
|    | robot      |   Batch-Size |   curobo-time(ms) |   curobo-succ |   curobo-pos-err |   curobo-ori-err |   pyroki-time(ms) |   pyroki-succ |   pyroki-pos-err(mm) |   pyroki-ori-err |
+====+============+==============+===================+===============+==================+==================+===================+===============+======================+==================+
|  0 | franka.yml |            1 |           4.48179 |        100    |      0.000739527 |      2.54939e-06 |           3.27039 |           100 |          9.5414e-05  |      2.52881e-07 |
+----+------------+--------------+-------------------+---------------+------------------+------------------+-------------------+---------------+----------------------+------------------+
|  1 | franka.yml |           10 |           4.71473 |        100    |      0.00267408  |      5.58381e-06 |           3.41916 |           100 |          0.000201231 |      2.30439e-07 |
+----+------------+--------------+-------------------+---------------+------------------+------------------+-------------------+---------------+----------------------+------------------+
|  2 | franka.yml |          100 |           6.31404 |        100    |      0.00359474  |      7.05379e-06 |           4.58956 |           100 |          0.000326011 |      3.99912e-07 |
+----+------------+--------------+-------------------+---------------+------------------+------------------+-------------------+---------------+----------------------+------------------+
|  3 | franka.yml |         1000 |          25.9619  |        100    |      0.00470231  |      6.96488e-06 |          14.1566  |           100 |          0.000303925 |      4.95823e-07 |
+----+------------+--------------+-------------------+---------------+------------------+------------------+-------------------+---------------+----------------------+------------------+
|  4 | franka.yml |         2000 |          50.2503  |         99.95 |      0.0043114   |      6.93035e-06 |          29.8455  |           100 |          0.000388033 |      4.88794e-07 |
+----+------------+--------------+-------------------+---------------+------------------+------------------+-------------------+---------------+----------------------+------------------+

Run with versions:
- Python 3.12
- curobo @ 0a50de1ba72db304195d59d9d0b1ed269696047f
- jaxls @ e43d482d747615323c23fb935bf215419ad07f1e
- jax 0.6.0, CUDA 12.4

Hardware:
- CPU: AMD Ryzen 7 7700X 8-Core Processor
- GPU: NVIDIA GeForce RTX 4090
"""


# pyright: reportMissingImports=false
# pyright: reportPossiblyUnboundVariable=false
# pyright: reportMissingModuleSource=false

#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Disable JAX prealloc etc
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Standard Library
import argparse
import time

# Third Party
import numpy as np
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    write_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# set seeds
torch.manual_seed(2)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


import xml.etree.ElementTree as ET
from io import StringIO

import jax
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy
from jax import lax
from jax import numpy as jnp
from robot_descriptions.loaders.yourdfpy import load_robot_description


def newton_raphson(f, x, iters):
    """Use the Newton-Raphson method to find a root of the given function."""

    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None

    x, _ = lax.scan(update, 1.0, length=iters)
    return x


def roberts_sequence(num_points, dim, root):
    # From https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab.
    basis = 1 - (1 / root ** (1 + jnp.arange(dim)))

    n = jnp.arange(num_points)
    x = n[:, None] * basis[None, :]
    x, _ = jnp.modf(x)

    return x


class PyrokiIkBeamHelper:
    def __init__(self):
        # Get the Panda robot. We fix the prismatic (gripper) joints. This is to
        # match cuRobo, it makes a very small runtime difference.
        urdf = load_robot_description("panda_description")
        xml_tree = urdf.write_xml()
        for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics"):
                child = joint.find(tag)
                if child is not None:
                    joint.remove(child)
        xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
        buf = StringIO(xml_str)
        urdf = yourdfpy.URDF.load(buf)
        assert urdf.validate()

        # yourdfpy => pyroki
        robot = pk.Robot.from_urdf(urdf)
        ee_link_name = "panda_hand_tcp"
        target_link_index = jnp.array(robot.links.names.index(ee_link_name))

        self.robot = robot
        exp = robot.joints.num_actuated_joints
        self.root = newton_raphson(lambda x: x ** (exp + 1) - x - 1, 1.0, 10_000)
        self.target_link_index = target_link_index

    def solve_ik(self, target_wxyz: jax.Array, target_position: jax.Array) -> jax.Array:
        num_seeds_init: int = 64
        num_seeds_final: int = 4

        total_steps: int = 16
        init_steps: int = 6

        def solve_one(
            initial_q: jax.Array, lambda_initial: float | jax.Array, max_iters: int
        ) -> tuple[jax.Array, jaxls.SolveSummary]:
            """Solve IK problem with a single initial condition. We'll vmap
            over initial_q to solve problems in parallel."""
            joint_var = robot.joint_var_cls(0)
            factors = [
                # pk.costs.pose_cost(
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    self.target_link_index,
                    pos_weight=10.0,
                    ori_weight=5.0,
                ),
                pk.costs.limit_cost(
                    robot,
                    joint_var,
                    weight=50.0,
                ),
            ]
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make(
                        [joint_var.with_value(initial_q)]
                    ),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # Create initial seeds, but this time with quasi-random sequence.
        robot = self.robot
        initial_qs = robot.joints.lower_limits + roberts_sequence(
            num_seeds_init, robot.joints.num_actuated_joints, self.root
        ) * (robot.joints.upper_limits - robot.joints.lower_limits)

        # Optimize the initial seeds.
        initial_sols, summary = vmapped_solve(
            initial_qs, jnp.full(initial_qs.shape[:1], 10.0), init_steps
        )

        # Get the best initial solutions.
        best_initial_sols = jnp.argsort(
            summary.cost_history[jnp.arange(num_seeds_init), -1]
        )[:num_seeds_final]

        # Optimize more for the best initial solutions.
        best_sols, summary = vmapped_solve(
            initial_sols[best_initial_sols],
            summary.lambda_history[jnp.arange(num_seeds_init), -1][best_initial_sols],
            total_steps - init_steps,
        )
        return best_sols[
            jnp.argmin(
                summary.cost_history[jnp.arange(num_seeds_final), summary.iterations]
            )
        ]

    def forward_kinematics(self, q: jax.Array | np.ndarray) -> jax.Array:
        return self.robot.forward_kinematics(jnp.asarray(q))[self.target_link_index]


# Batched helpers for IK and FK.
ik_beam = PyrokiIkBeamHelper()
batched_ik = jax.jit(jax.vmap(ik_beam.solve_ik))
batched_fk = jax.jit(jax.vmap(ik_beam.forward_kinematics))


def evaluate_pyroki_ik(q_sample: torch.Tensor):
    # Get target poses.
    q_sample = q_sample.numpy(force=True)  # type: ignore
    target_wxyz_xyz = batched_fk(q_sample)
    target_wxyz_jax = target_wxyz_xyz[..., 0:4]
    target_position_jax = target_wxyz_xyz[..., 4:7]

    # JIT compile
    jax.block_until_ready(batched_ik(target_wxyz_jax, target_position_jax))

    # Run the function
    start = time.time()
    solution = batched_ik(target_wxyz_jax, target_position_jax)
    jax.block_until_ready(solution)
    total_time = (time.time() - start) / target_wxyz_jax.shape[0]

    # Do FK
    fk_result = batched_fk(solution)
    assert fk_result.shape == (target_wxyz_jax.shape[0], 7)
    position_error = np.linalg.norm(
        np.array(fk_result[:, 4:7]) - np.array(target_position_jax),
        axis=-1,
    )
    assert position_error.shape == (target_wxyz_jax.shape[0],)

    # Copied from cuRobo
    position_threshold: float = 0.005
    rotation_threshold: float = 0.05

    orientation_error = np.linalg.norm(
        np.array(
            (
                jaxlie.SO3(target_wxyz_jax).inverse() @ jaxlie.SO3(fk_result[:, 0:4])
            ).log()
        ),
        axis=-1,
    )
    assert orientation_error.shape == (target_wxyz_jax.shape[0],)

    success_mask = np.logical_and(
        position_error < position_threshold,
        orientation_error < rotation_threshold,
    )
    return (
        total_time,
        np.mean(success_mask) * 100.0,
        np.percentile(position_error[success_mask], 98),
        np.percentile(orientation_error[success_mask], 98),
    )


def run_full_config_collision_free_ik(
    robot_file,
    world_file,
    batch_size,
    use_cuda_graph=False,
    collision_free=True,
    high_precision=False,
    num_seeds=12,
):
    tensor_args = TensorDeviceType()
    robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    if not collision_free:
        robot_data["kinematics"]["collision_link_names"] = None
        robot_data["kinematics"]["lock_joints"] = {}
    robot_data["kinematics"]["collision_sphere_buffer"] = 0.0
    robot_cfg = RobotConfig.from_dict(robot_data)
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), world_file))
    )
    position_threshold = 0.005
    grad_iters = None
    if high_precision:
        position_threshold = 0.001
        grad_iters = 100
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        position_threshold=position_threshold,
        num_seeds=num_seeds,
        self_collision_check=collision_free,
        self_collision_opt=collision_free,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
        high_precision=high_precision,
        regularization=False,
        grad_iters=grad_iters,
    )
    ik_solver = IKSolver(ik_config)

    for i in range(3):
        q_sample = ik_solver.sample_configs(batch_size)
        while q_sample.shape[0] == 0:
            q_sample = ik_solver.sample_configs(batch_size)

        kin_state = ik_solver.fk(q_sample)
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        total_time = (time.time() - st_time) / q_sample.shape[0]
        if i == 0:
            pyroki_out = evaluate_pyroki_ik(q_sample)

    curobo_out = (
        total_time,
        100.0 * torch.count_nonzero(result.success).item() / len(q_sample),
        # np.mean(result.position_error[result.success].cpu().numpy()).item(),
        np.percentile(result.position_error[result.success].cpu().numpy(), 98).item(),
        np.percentile(result.rotation_error[result.success].cpu().numpy(), 98).item(),
    )

    return curobo_out, pyroki_out


if __name__ == "__main__":
    setup_curobo_logger("error")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="path to save file",
    )
    parser.add_argument(
        "--high_precision",
        action="store_true",
        help="When True, enables IK for 1 mm precision, when False 5mm precision",
        default=False,
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="ik",
        help="File name prefix to use to save benchmark results",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=16,
        help="Number of seeds to use for IK",
    )
    args = parser.parse_args()

    b_list = [1, 10, 100, 1000, 2000]

    robot_list = ["franka.yml"]  # get_motion_gen_robot_list()
    world_file = "collision_test.yml"

    print("running...")

    from collections import defaultdict

    data = defaultdict(list)
    for robot_file in robot_list:
        print("running for robot: ", robot_file)
        # create a sampler with dof:
        for b_size in b_list:
            # sample test configs:
            (
                (dt_cu_ik, succ, p_err, q_err),
                (pyroki_dt, pyroki_succ, pyroki_p_err, pyroki_q_err),
            ) = run_full_config_collision_free_ik(
                robot_file,
                world_file,
                batch_size=b_size,
                use_cuda_graph=True,
                collision_free=False,
                high_precision=args.high_precision,
                num_seeds=args.num_seeds,
            )
            data["robot"].append(robot_file)
            data["Batch-Size"].append(b_size)

            data["curobo-time(ms)"].append(dt_cu_ik * 1000.0 * b_size)
            data["curobo-succ"].append(succ)
            data["curobo-pos-err"].append(p_err * 1000.0)
            data["curobo-ori-err"].append(q_err)

            data["pyroki-time(ms)"].append(pyroki_dt * 1000.0 * b_size)
            data["pyroki-succ"].append(pyroki_succ)
            data["pyroki-pos-err(mm)"].append(pyroki_p_err * 1000.0)
            data["pyroki-ori-err"].append(pyroki_q_err)

    if args.save_path is not None:
        file_path = join_path(args.save_path, args.file_name)
    else:
        file_path = args.file_name

    write_yaml(data, file_path + ".yml")

    try:
        # Third Party
        import pandas as pd

        df = pd.DataFrame(data)
        print("Reported errors are 98th percentile")
        df.to_csv(file_path + ".csv")
        try:
            # Third Party
            from tabulate import tabulate

            print(tabulate(df, headers="keys", tablefmt="grid"))
        except ImportError:
            print(df)

            pass
    except ImportError:
        pass
