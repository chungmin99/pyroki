"""URDF parsing utilities for Robot class."""

from copy import deepcopy

import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import yourdfpy
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int
from loguru import logger


@jdc.pytree_dataclass
class JointInfo:
    """Contains joint-related information for a robot."""

    num_joints: jdc.Static[int]
    num_actuated_joints: jdc.Static[int]
    names: jdc.Static[tuple[str, ...]]
    actuated_names: jdc.Static[tuple[str, ...]]

    # Joint parameters.
    twists: Float[Array, "n_joints 6"]
    """Twist parameters for each joint. Shape: (n_joints, 6)."""
    parent_transforms: Float[Array, "n_joints 7"]
    """Transform from parent joint to current joint. Shape: (n_joints, 7)."""
    parent_indices: Int[Array, " n_joints"]
    """Index of the parent joint for each joint. Shape: (n_joints,)."""
    actuated_indices: Int[Array, " n_joints"]
    """Index of the associated actuated joint that drives each joint. -1 if mimic joint. Shape: (n_joints,)."""

    # Limits for actuated joints.
    lower_limits: Float[Array, " n_act_joints"]
    """Lower limits for directly actuated joints. Shape: (n_act_joints,)."""
    upper_limits: Float[Array, " n_act_joints"]
    """Upper limits for directly actuated joints. Shape: (n_act_joints,)."""
    velocity_limits: Float[Array, " n_act_joints"]
    """Velocity limits for directly actuated joints. Shape: (n_act_joints,)."""

    # Effective limits for ALL joints (including mimics and fixed).
    lower_limits_all: Float[Array, " n_joints"]
    """Lower limits for all joints. Shape: (n_joints,)."""
    upper_limits_all: Float[Array, " n_joints"]
    """Upper limits for all joints. Shape: (n_joints,)."""
    velocity_limits_all: Float[Array, " n_joints"]
    """Velocity limits for all joints. Shape: (n_joints,)."""

    # Mimic joint parameters.
    mimic_multiplier: Float[Array, " n_joints"]
    """Mimic multiplier for each joint. Shape: (n_joints,). 1.0 if not a mimic joint."""
    mimic_offset: Float[Array, " n_joints"]
    """Mimic offset for each joint. Shape: (n_joints,). 0 if not a mimic joint."""
    mimic_act_indices: Int[Array, " n_joints"]
    """Index of the actuated joint that is mimicked by each joint. -1 if not a mimic joint. Shape: (n_joints,)."""

    _topo_sort_inv: Int[Array, " n_joints"]
    """Inverse topological sort order, mapping sorted joint index to original joint index."""

    def _map_to_full_joint_space(
        self,
        value_actuated: Float[Array, "*batch n_act_joints"],
        apply_offset: bool,
    ) -> Float[Array, "*batch n_joints"]:
        """Internal helper to map actuated values to full joint space,
        optionally applying mimic offset."""
        batch_axes = value_actuated.shape[:-1]
        assert value_actuated.shape == (*batch_axes, self.num_actuated_joints)

        # Pad input vector to handle fixed joints (index -1) safely.
        value_padded = jnp.concatenate(
            [value_actuated, jnp.zeros((*batch_axes, 1))], axis=-1
        )

        # Replace mimic indices with the actuated joint index they refer to.
        replace_mimic_indices = jnp.where(
            self.mimic_act_indices == -1,
            self.actuated_indices,
            self.mimic_act_indices,
        )

        safe_actuated_indices = jnp.where(
            replace_mimic_indices == -1,
            self.num_actuated_joints,  # Point to the zero padding.
            replace_mimic_indices,
        )

        # value_referenced contains the value of the joint each joint index refers to.
        value_referenced = value_padded[..., safe_actuated_indices]
        assert value_referenced.shape == (*batch_axes, self.num_joints)

        # Apply mimic multiplier.
        value_multiplied = value_referenced * self.mimic_multiplier

        # Conditionally add mimic offset.
        value_full = jnp.where(
            apply_offset, value_multiplied + self.mimic_offset, value_multiplied
        )
        assert value_full.shape == (*batch_axes, self.num_joints)

        return value_full

    def get_full_config(
        self,
        cfg_actuated: Float[Array, "*batch n_act_joints"],
    ) -> Float[Array, "*batch n_joints"]:
        """Compute the full joint configuration vector (for all n_joints)
        from the configuration of the actuated joints (n_act_joints).

        Handles fixed joints and applies mimic joint relationships (multiplier + offset).

        Args:
            cfg_actuated: Configuration of the actuated joints.

        Returns:
            Full configuration vector.
        """
        return self._map_to_full_joint_space(cfg_actuated, apply_offset=True)


@jdc.pytree_dataclass
class LinkInfo:
    """Contains link-related information for a robot."""

    num_links: jdc.Static[int]
    names: jdc.Static[tuple[str, ...]]
    parent_joint_indices: Int[Array, "n_links"]


class RobotURDFParser:
    """Parser for creating Robot instances from URDF files."""

    @staticmethod
    def _topologically_sort_joints(
        urdf: yourdfpy.URDF,
    ) -> Int[Array, " joints"]:
        """Calculates the topological processing order for joints and actuated joints.

        Ensures joints are processed parent-first for kinematic calculations, respecting
        mimic joint dependencies.

        Returns:
            - joint_order: Array of original joint indices sorted topologically.
        """
        original_joints = list(urdf.joint_map.values())
        num_joints = len(original_joints)
        original_name_to_idx = {j.name: i for i, j in enumerate(original_joints)}

        # Perform topological sort based on parent-child and mimic relationships.
        joints_to_sort = deepcopy(original_joints)
        sorted_joint_objects = list[yourdfpy.Joint]()
        parent_link_of_joint = {j.child: j.parent for j in joints_to_sort}
        child_link_of_joint = {j.name: j.child for j in joints_to_sort}
        mimic_map = {
            j.name: j.mimic.joint for j in joints_to_sort if j.mimic is not None
        }

        processed_child_links = set()
        processed_joint_names = set()

        while len(sorted_joint_objects) < num_joints:
            found_next = False
            for i, j in enumerate(joints_to_sort):
                parent_link = parent_link_of_joint.get(j.child)

                # Check if parent link is ready
                parent_ok = (
                    parent_link == urdf.base_link
                    or parent_link in processed_child_links
                )
                # Check if mimic dependency is met
                mimic_ok = (j.name not in mimic_map) or (
                    mimic_map[j.name] in processed_joint_names
                )

                if parent_ok and mimic_ok:
                    sorted_joint_objects.append(j)
                    processed_child_links.add(child_link_of_joint[j.name])
                    processed_joint_names.add(j.name)
                    joints_to_sort.pop(i)
                    found_next = True
                    break
            if not found_next:
                # Simplified error handling for brevity during refactor
                remaining_names = [j.name for j in joints_to_sort]
                raise ValueError(
                    f"Topological sort failed. Remaining: {remaining_names}"
                )

        # Generate the topological order based on original indices
        joint_order = jnp.array(
            [original_name_to_idx[j.name] for j in sorted_joint_objects],
            dtype=jnp.int32,
        )

        if jnp.any(joint_order != jnp.arange(num_joints)):
            logger.info(
                "Joints were not in topological order; they will be internally sorted."
            )

        return joint_order

    @staticmethod
    def parse(urdf: yourdfpy.URDF) -> tuple[JointInfo, LinkInfo]:
        """Build joint and link information from a URDF."""
        joint_twists_list = list[Array]()
        parent_transform_list = list[Array]()
        parent_idx_list = list[int]()
        actuated_idx_list = list[int]()
        lower_limit_act_list = list[float]()
        upper_limit_act_list = list[float]()
        velocity_limit_act_list = list[float]()
        joint_name_list = list[str]()
        actuated_name_list = list[str]()
        mimic_multiplier_list = list[float]()
        mimic_offset_list = list[float]()
        mimic_act_idx_list = list[int]()

        # Store limits read directly from URDF for *all* joints -> _eff
        lower_limit_eff_list = list[float]()
        upper_limit_eff_list = list[float]()
        velocity_limit_eff_list = list[float]()

        # Link information.
        link_name_list = list[str]()
        parent_joint_idx_list = list[int]()

        # First pass: collect joint information.
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            # Get joint names.
            joint_name_list.append(joint.name)

            # Get twist for *this* joint (will be zero for fixed)
            twist = RobotURDFParser._get_joint_twist(joint)
            joint_twists_list.append(twist)

            # Get the actuated joint index it refers to (could be itself or mimicked).
            # Also get mimic parameters if applicable.
            act_idx, mimic_act_idx, multiplier, offset = (
                RobotURDFParser._get_act_joint_idx_and_mimic(urdf, joint)
            )
            actuated_idx_list.append(act_idx)
            mimic_act_idx_list.append(mimic_act_idx)
            mimic_multiplier_list.append(multiplier)
            mimic_offset_list.append(offset)
            if act_idx != -1:
                actuated_name_list.append(joint.name)

            # Get effective joint limits.
            lower_eff, upper_eff = RobotURDFParser._get_joint_limits(joint)
            vel_limit_eff = RobotURDFParser._get_joint_limit_vel(joint)

            lower_limit_eff_list.append(lower_eff)
            upper_limit_eff_list.append(upper_eff)
            velocity_limit_eff_list.append(vel_limit_eff)

            # Get directly actuated joint limits.
            if joint in urdf.actuated_joints and mimic_act_idx == -1:
                lower_limit_act_list.append(lower_eff)
                upper_limit_act_list.append(upper_eff)
                velocity_limit_act_list.append(vel_limit_eff)

            # Get the parent joint index and transform for each joint.
            parent_idx, T_parent_joint_val = RobotURDFParser._get_T_parent_joint(
                urdf, joint
            )
            parent_idx_list.append(parent_idx)
            parent_transform_list.append(T_parent_joint_val)

        # Second pass: collect link information.
        joint_from_link = {j.child: j for j in urdf.joint_map.values()}
        for link_name in urdf.link_map:
            link_name_list.append(link_name)
            if link_name in joint_from_link:
                joint_idx = joint_name_list.index(joint_from_link[link_name].name)
                parent_joint_idx_list.append(joint_idx)
            else:
                parent_joint_idx_list.append(-1)

        # Calculate topological sort order
        topo_sort_inv_val = RobotURDFParser._topologically_sort_joints(urdf)

        # Convert collected lists to arrays
        lower_limits_act_arr = jnp.array(lower_limit_act_list)
        upper_limits_act_arr = jnp.array(upper_limit_act_list)
        velocity_limits_act_arr = jnp.array(velocity_limit_act_list)
        actuated_indices_arr = jnp.array(actuated_idx_list, dtype=jnp.int32)
        mimic_multiplier_arr = jnp.array(mimic_multiplier_list)
        mimic_offset_arr = jnp.array(mimic_offset_list)
        lower_limits_eff_arr = jnp.array(lower_limit_eff_list)
        upper_limits_eff_arr = jnp.array(upper_limit_eff_list)
        velocity_limits_eff_arr = jnp.array(velocity_limit_eff_list)

        # Create JointInfo and LinkInfo based on original order.
        joint_info = JointInfo(
            num_joints=len(urdf.joint_map),
            num_actuated_joints=len(urdf.actuated_joints),
            twists=jnp.array(joint_twists_list),
            parent_transforms=jnp.array(parent_transform_list),
            parent_indices=jnp.array(parent_idx_list, dtype=jnp.int32),
            actuated_indices=actuated_indices_arr,
            lower_limits=lower_limits_act_arr,
            upper_limits=upper_limits_act_arr,
            velocity_limits=velocity_limits_act_arr,
            names=tuple(joint_name_list),
            actuated_names=tuple(actuated_name_list),
            lower_limits_all=lower_limits_eff_arr,
            upper_limits_all=upper_limits_eff_arr,
            velocity_limits_all=velocity_limits_eff_arr,
            mimic_multiplier=mimic_multiplier_arr,
            mimic_offset=mimic_offset_arr,
            mimic_act_indices=jnp.array(mimic_act_idx_list),
            _topo_sort_inv=topo_sort_inv_val,
        )
        assert joint_info.twists.shape == (joint_info.num_joints, 6)
        assert joint_info.parent_transforms.shape == (joint_info.num_joints, 7)
        assert joint_info.parent_indices.shape == (joint_info.num_joints,)
        assert joint_info.actuated_indices.shape == (joint_info.num_joints,)
        assert joint_info.lower_limits.shape == (joint_info.num_actuated_joints,)
        assert joint_info.upper_limits.shape == (joint_info.num_actuated_joints,)
        assert joint_info.velocity_limits.shape == (joint_info.num_actuated_joints,)
        assert joint_info.lower_limits_all.shape == (joint_info.num_joints,)
        assert joint_info.upper_limits_all.shape == (joint_info.num_joints,)
        assert joint_info.velocity_limits_all.shape == (joint_info.num_joints,)
        assert joint_info._topo_sort_inv.shape == (joint_info.num_joints,)
        assert joint_info.mimic_multiplier.shape == (joint_info.num_joints,)
        assert joint_info.mimic_offset.shape == (joint_info.num_joints,)
        assert joint_info.mimic_act_indices.shape == (joint_info.num_joints,)

        link_info = LinkInfo(
            num_links=len(link_name_list),
            names=tuple(link_name_list),
            parent_joint_indices=jnp.array(parent_joint_idx_list, dtype=jnp.int32),
        )
        assert link_info.parent_joint_indices.shape == (link_info.num_links,)
        return joint_info, link_info

    @staticmethod
    def _get_act_joint_idx_and_mimic(
        urdf: yourdfpy.URDF, joint: yourdfpy.Joint
    ) -> tuple[int, int, float, float]:
        """Get the index of the actuated joint for a joint, and mimic parameters."""
        mimic_act_idx = -1
        multiplier = 1.0
        offset = 0.0

        # Check if this joint is a mimic joint.
        if joint.mimic is not None:
            if joint.mimic.multiplier is not None:
                multiplier = joint.mimic.multiplier
            if joint.mimic.offset is not None:
                offset = joint.mimic.offset

            act_joint_idx = -1
            mimicked_joint_name = joint.mimic.joint
            mimicked_joint = urdf.joint_map[mimicked_joint_name]
            mimic_act_idx = urdf.actuated_joints.index(mimicked_joint)

        # If not mimic, check if it's directly actuated.
        elif joint in urdf.actuated_joints:
            assert joint.axis.shape == (3,)
            act_joint_idx = urdf.actuated_joints.index(joint)

        # Otherwise, it's a fixed joint.
        else:
            act_joint_idx = -1  # Represents non-actuated/fixed.

        return act_joint_idx, mimic_act_idx, multiplier, offset

    @staticmethod
    def _get_joint_twist(joint: yourdfpy.Joint) -> Array:
        """Get the twist parameters for any joint (zero for fixed)."""
        if joint.type in ("revolute", "continuous"):
            twist = jnp.concatenate([jnp.zeros(3), joint.axis])
        elif joint.type == "prismatic":
            twist = jnp.concatenate([joint.axis, jnp.zeros(3)])
        elif joint.type == "fixed":
            twist = jnp.zeros(6)
        else:
            # Floating joints etc. are not supported yet.
            logger.warning(
                f"Unsupported joint type {joint.type} encountered for joint '{joint.name}'. Treating as fixed."
            )
            twist = jnp.zeros(6)
            # raise ValueError(f"Unsupported joint type {joint.type}!")
        assert twist.shape == (6,)
        return twist

    @staticmethod
    def _get_T_parent_joint(
        urdf: yourdfpy.URDF,
        joint: yourdfpy.Joint,
    ) -> tuple[int, Array]:
        """Get the transform from the parent joint to the current joint,
        as well as the parent joint index."""
        # Handle case where joint origin might be None (e.g., base link)
        if joint.origin is None:
            logger.debug(
                f"Joint '{joint.name}' has no origin, assuming identity transform."
            )
            T_parent_joint = onp.eye(4)
        else:
            assert joint.origin.shape == (4, 4)
            T_parent_joint = joint.origin

        joint_from_child = {j.child: j for j in urdf.joint_map.values()}
        joint_name_to_idx = {j.name: i for i, j in enumerate(urdf.joint_map.values())}

        if joint.parent not in joint_from_child:
            # Must be root node's joint (parent is base_link).
            parent_index = -1
        else:
            parent_joint = joint_from_child[joint.parent]
            parent_index = joint_name_to_idx[parent_joint.name]

        return (parent_index, jaxlie.SE3.from_matrix(T_parent_joint).wxyz_xyz)

    @staticmethod
    def _get_joint_limits(joint: yourdfpy.Joint) -> tuple[float, float]:
        """Get the joint limits defined in the URDF.
        Assumes caller checked that joint.limit.lower/upper exist OR type is fixed/continuous.
        """
        if joint.type == "fixed":
            return 0.0, 0.0
        elif joint.type == "continuous":
            if (
                joint.limit is not None
                and joint.limit.lower is not None
                and joint.limit.upper is not None
            ):
                return joint.limit.lower, joint.limit.upper
            else:
                logger.warning(
                    f"Continuous joint '{joint.name}' has no explicit limits. "
                    "Returning [-pi, pi]."
                )
                return -jnp.pi, jnp.pi
        elif (
            joint.limit is not None
            and joint.limit.lower is not None
            and joint.limit.upper is not None
        ):
            return joint.limit.lower, joint.limit.upper
        elif joint.type == "fixed":
            return 0.0, 0.0
        else:
            raise ValueError(f"Joint '{joint.name}' ({joint.type}) has no limits.")

    @staticmethod
    def _get_joint_limit_vel(joint: yourdfpy.Joint) -> float:
        """Get the joint velocity limit defined in the URDF.
        Assumes checks for existence have been done prior to calling, or joint is fixed.
        """
        if joint.type == "fixed":
            return 0.0
        elif joint.limit is not None and joint.limit.velocity is not None:
            return joint.limit.velocity
        else:
            raise ValueError(
                f"Joint '{joint.name}' of type '{joint.type}' has no velocity limits."
            )
