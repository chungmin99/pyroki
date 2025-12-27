from __future__ import annotations

import jax
import jax_dataclasses as jdc
import jaxlie
import jaxls
import yourdfpy
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from ._robot_urdf_parser import JointInfo, LinkInfo, RobotURDFParser


@jdc.pytree_dataclass
class Robot:
    """A differentiable robot kinematics tree."""

    joints: JointInfo
    """Joint information for the robot."""

    links: LinkInfo
    """Link information for the robot."""

    joint_var_cls: jdc.Static[type[jaxls.Var[Array]]]
    """Variable class for the robot configuration."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        default_joint_cfg: Float[ArrayLike, "*batch actuated_count"] | None = None,
    ) -> Robot:
        """
        Loads a robot kinematic tree from a URDF.
        Internally tracks a topological sort of the joints.

        Args:
            urdf: The URDF to load the robot from.
            default_joint_cfg: The default joint configuration to use for optimization.
        """
        joints, links = RobotURDFParser.parse(urdf)

        # Compute default joint configuration.
        if default_joint_cfg is None:
            default_joint_cfg = (joints.lower_limits + joints.upper_limits) / 2
        else:
            default_joint_cfg = jnp.array(default_joint_cfg)
        assert default_joint_cfg.shape == (joints.num_actuated_joints,)

        # Variable class for the robot configuration.
        class JointVar(  # pylint: disable=missing-class-docstring
            jaxls.Var[Array],
            default_factory=lambda: default_joint_cfg,
        ): ...

        robot = Robot(
            joints=joints,
            links=links,
            joint_var_cls=JointVar,
        )

        return robot

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch actuated_count"],
        unroll_fk: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch link_count 7"]:
        """Run forward kinematics on the robot's links, in the provided configuration.

        Computes the world pose of each link frame. The result is ordered
        corresponding to `self.link.names`.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch actuated_count)`.

        Returns:
            The SE(3) transforms of the links, ordered by `self.link.names`,
            in the format `(*batch, link_count, wxyz_xyz)`.
        """
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.joints.num_actuated_joints)
        return self._link_poses_from_joint_poses(
            self._forward_kinematics_joints(cfg, unroll_fk)
        )

    def _link_poses_from_joint_poses(
        self, Ts_world_joint: Float[Array, "*batch actuated_count 7"]
    ) -> Float[Array, "*batch link_count 7"]:
        (*batch_axes, _, _) = Ts_world_joint.shape
        # Get the link poses.
        # self.links.parent_joint_indices is Static[tuple], cast to array for indexing
        link_parent_indices = jnp.array(
            self.links.parent_joint_indices, dtype=jnp.int32
        )

        base_link_mask = link_parent_indices == -1
        parent_joint_indices = jnp.where(base_link_mask, 0, link_parent_indices)
        identity_pose = jaxlie.SE3.identity().wxyz_xyz
        Ts_world_link = jnp.where(
            base_link_mask[..., None],
            identity_pose,
            Ts_world_joint[..., parent_joint_indices, :],
        )
        assert Ts_world_link.shape == (*batch_axes, self.links.num_links, 7)
        return Ts_world_link

    def _forward_kinematics_joints(
        self,
        cfg: Float[Array, "*batch actuated_count"],
        unroll_fk: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch joint_count 7"]:
        (*batch_axes, _) = cfg.shape
        assert cfg.shape == (*batch_axes, self.joints.num_actuated_joints)

        # Calculate full configuration using the dedicated method
        q_full = self.joints.get_full_config(cfg)

        # Calculate delta transforms using the effective config and twists for all joints.
        tangents = self.joints.twists * q_full[..., None]
        assert tangents.shape == (*batch_axes, self.joints.num_joints, 6)
        delta_Ts = jaxlie.SE3.exp(tangents)  # Shape: (*batch_axes, self.joint.count, 7)

        # Combine constant parent transform with variable joint delta transform.
        Ts_parent_child = (
            jaxlie.SE3(self.joints.parent_transforms) @ delta_Ts
        ).wxyz_xyz
        assert Ts_parent_child.shape == (*batch_axes, self.joints.num_joints, 7)

        # Topological sort helpers
        # Static tuples must be cast to arrays for advanced indexing/argsort
        topo_sort_inv = jnp.array(self.joints._topo_sort_inv, dtype=jnp.int32)
        topo_order = jnp.argsort(topo_sort_inv)

        Ts_parent_child_sorted = Ts_parent_child[..., topo_sort_inv, :]

        parent_indices = jnp.array(self.joints.parent_indices, dtype=jnp.int32)
        parent_orig_for_sorted_child = parent_indices[topo_sort_inv]
        idx_parent_joint_sorted = jnp.where(
            parent_orig_for_sorted_child == -1,
            -1,
            topo_order[parent_orig_for_sorted_child],
        )

        # Compute link transforms relative to world, indexed by sorted *joint* index.
        def compute_transform(i: int, Ts_world_link_sorted: Array) -> Array:
            parent_sorted_idx = idx_parent_joint_sorted[i]
            T_world_parent_link = jnp.where(
                parent_sorted_idx == -1,
                jaxlie.SE3.identity().wxyz_xyz,
                Ts_world_link_sorted[..., parent_sorted_idx, :],
            )
            return Ts_world_link_sorted.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent_link)
                    @ jaxlie.SE3(Ts_parent_child_sorted[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_link_init_sorted = jnp.zeros((*batch_axes, self.joints.num_joints, 7))
        Ts_world_link_sorted = jax.lax.fori_loop(
            lower=0,
            upper=self.joints.num_joints,
            body_fun=compute_transform,
            init_val=Ts_world_link_init_sorted,
            unroll=unroll_fk,
        )

        Ts_world_link_joint_indexed = Ts_world_link_sorted[..., topo_order, :]
        assert Ts_world_link_joint_indexed.shape == (
            *batch_axes,
            self.joints.num_joints,
            7,
        )  # This is the link poses indexed by parent *joint* index.

        return Ts_world_link_joint_indexed
