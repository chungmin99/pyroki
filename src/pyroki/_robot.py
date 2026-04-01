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

    @staticmethod
    def from_reduced_urdf(
        input_urdf_path: str, 
        preserved_joint_names: list[str], 
        output_urdf_path: str,
        default_joint_cfg: Float[ArrayLike, "*batch actuated_count"] | None = None,
    ) -> Robot:
        """
        Create a new URDF file with only preserved_joint_names.
        Note: This function preserves the root joint as root, and connects preserved_joint_names to the root.
        
        Args:
            input_urdf_path: Path to the original full URDF file.
            preserved_joint_names: List of joint names to preserve.
            output_urdf_path: Path to the output reduced URDF file.
            default_joint_cfg: Joint configuration for the reduced urdf model.
        
        Returns:
            The reduced URDF as a yourdfpy.URDF object.
        """
        import os
        from lxml import etree

        if os.path.exists(output_urdf_path):
            print(f"Output file {output_urdf_path} already exists.")
            print(f"Loading existing URDF from {output_urdf_path}.")
            urdf = yourdfpy.URDF.load(output_urdf_path)
            print(f"Loaded URDF from {output_urdf_path}: {urdf.__dict__}")
            return Robot.from_urdf(urdf, default_joint_cfg)
            
        # Parse original URDF
        tree = etree.parse(input_urdf_path)
        root = tree.getroot()
        
        # Load to get joint structure
        urdf = yourdfpy.URDF.load(input_urdf_path)
        
        # Find required elements
        required_links = set()
        required_joints = set(preserved_joint_names)
        
        for joint_name in preserved_joint_names:
            if joint_name in urdf.joint_map:
                joint = urdf.joint_map[joint_name]
                required_links.add(joint.child)
                
                # Traverse to root
                current = joint.parent
                while current:
                    required_links.add(current)
                    parent_joint = next((j for j in urdf.joint_map.values() if j.child == current), None)
                    if parent_joint:
                        required_joints.add(parent_joint.name)
                        current = parent_joint.parent
                    else:
                        break
        
        print(f"Required joints: {len(required_joints)}")
        print(f"Required links: {len(required_links)}")
        
        # Remove unwanted elements from XML
        for joint in list(root.findall('joint')):
            if joint.get('name') not in required_joints:
                root.remove(joint)
        
        for link in list(root.findall('link')):
            if link.get('name') not in required_links:
                root.remove(link)
        
        # VALIDATE: Check that all joint parents/children exist
        for joint_elem in root.findall('joint'):
            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            
            parent_link = parent_elem.get('link') if parent_elem is not None else None
            child_link = child_elem.get('link') if child_elem is not None else None
            
            if parent_link and parent_link not in required_links:
                print(f"WARNING: Joint {joint_elem.get('name')} references missing parent link: {parent_link}")
            if child_link and child_link not in required_links:
                print(f"WARNING: Joint {joint_elem.get('name')} references missing child link: {child_link}")
        
        # Save filtered URDF with explicit encoding
        tree.write(output_urdf_path, pretty_print=True, xml_declaration=True, encoding='utf-8')
        print(f"Saved reduced URDF to {output_urdf_path}")
        print(f"  Joints: {len(urdf.joint_map)} -> {len(required_joints)}")
        print(f"  Links: {len(urdf.link_map)} -> {len(required_links)}")
        
        # VERIFY: Try loading the reduced URDF with yourdfpy
        try:
            reduced_urdf = yourdfpy.URDF.load(output_urdf_path)
            print(f"✓ Reduced URDF loads successfully with yourdfpy")
            print(f"  Joints in reduced: {len(reduced_urdf.joint_map)}")
            print(f"  Links in reduced: {len(reduced_urdf.link_map)}")
        except Exception as e:
            print(f"✗ ERROR loading reduced URDF: {e}")
        else:
            return Robot.from_urdf(reduced_urdf, default_joint_cfg)


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
