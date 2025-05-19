from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import trimesh
import yourdfpy
from jaxtyping import Array, Float, Int
from loguru import logger

if TYPE_CHECKING:
    from pyroki._robot import Robot

from .._robot_urdf_parser import RobotURDFParser
from ._collision import collide, pairwise_collide
from ._geometry import Capsule, CollGeom


@jdc.pytree_dataclass
class RobotCollision:
    """Collision model for a robot, integrated with pyroki kinematics."""

    num_links: jdc.Static[int]
    """Number of links in the model (matches kinematics links)."""
    link_names: jdc.Static[tuple[str, ...]]
    """Names of the links corresponding to link indices."""
    coll: CollGeom
    """Collision geometries for the robot (relative to their parent link frame)."""

    active_idx_i: Int[Array, " P"]
    """Row indices (first link) of active self-collision pairs to check."""
    active_idx_j: Int[Array, " P"]
    """Column indices (second link) of active self-collision pairs to check."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...] = (),
        ignore_immediate_adjacents: bool = True,
    ):
        """
        Build a differentiable robot collision model from a URDF.

        Args:
            urdf: The URDF object (used to load collision meshes).
            user_ignore_pairs: Additional pairs of link names to ignore for self-collision.
            ignore_immediate_adjacents: If True, automatically ignore collisions
                between adjacent (parent/child) links based on the URDF structure.
        """
        # Re-load urdf with collision data if not already loaded.
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        try:
            has_collision = any(link.collisions for link in urdf.link_map.values())
            if not has_collision:
                urdf = yourdfpy.URDF(
                    robot=urdf.robot,
                    filename_handler=filename_handler,
                    load_collision_meshes=True,
                )
        except Exception as e:
            logger.warning(f"Could not reload URDF with collision meshes: {e}")

        _, link_info = RobotURDFParser.parse(urdf)
        link_name_list = link_info.names  # Use names from parser

        # Gather all collision meshes.
        # The order of cap_list must match link_name_list.
        cap_list = list[Capsule]()
        for link_name in link_name_list:
            cap_list.append(
                Capsule.from_trimesh(
                    RobotCollision._get_trimesh_collision_geometries(urdf, link_name)
                )
            )

        # Convert list of trimesh objects into a batched Capsule object.
        capsules = cast(Capsule, jax.tree.map(lambda *args: jnp.stack(args), *cap_list))
        assert capsules.get_batch_axes() == (link_info.num_links,)

        # Directly compute active pair indices
        active_idx_i, active_idx_j = RobotCollision._compute_active_pair_indices(
            link_names=link_name_list,
            urdf=urdf,
            user_ignore_pairs=user_ignore_pairs,
            ignore_immediate_adjacents=ignore_immediate_adjacents,
        )

        logger.info(
            f"Created RobotCollision with {link_info.num_links} links and "
            f"{len(active_idx_i)} active self-collision pairs."
        )

        return RobotCollision(
            num_links=link_info.num_links,
            link_names=link_name_list,
            active_idx_i=active_idx_i,
            active_idx_j=active_idx_j,
            coll=capsules,
        )

    @staticmethod
    def _compute_active_pair_indices(
        link_names: tuple[str, ...],
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
    ) -> Tuple[Int[Array, " P"], Int[Array, " P"]]:
        """
        Computes the indices (i, j) of pairs where i < j and the pair should
        be actively checked for self-collision.

        Args:
            link_names: Tuple of link names in order.
            urdf: Parsed URDF object.
            user_ignore_pairs: List of (name1, name2) pairs to explicitly ignore.
            ignore_immediate_adjacents: Whether to ignore parent-child pairs from URDF.

        Returns:
            Tuple of (active_i, active_j) index arrays.
        """
        # --- Start: Logic combined from _build_ignore_matrix --- #
        num_links = len(link_names)
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}
        ignore_matrix = jnp.zeros((num_links, num_links), dtype=bool)
        ignore_matrix = ignore_matrix.at[
            jnp.arange(num_links), jnp.arange(num_links)
        ].set(True)
        if ignore_immediate_adjacents:
            for joint in urdf.joint_map.values():
                parent_name = joint.parent
                child_name = joint.child
                if parent_name in link_name_to_idx and child_name in link_name_to_idx:
                    parent_idx = link_name_to_idx[parent_name]
                    child_idx = link_name_to_idx[child_name]
                    ignore_matrix = ignore_matrix.at[parent_idx, child_idx].set(True)
                    ignore_matrix = ignore_matrix.at[child_idx, parent_idx].set(True)
        for name1, name2 in user_ignore_pairs:
            if name1 in link_name_to_idx and name2 in link_name_to_idx:
                idx1 = link_name_to_idx[name1]
                idx2 = link_name_to_idx[name2]
                ignore_matrix = ignore_matrix.at[idx1, idx2].set(True)
                ignore_matrix = ignore_matrix.at[idx2, idx1].set(True)

        idx_i, idx_j = jnp.tril_indices(num_links, k=-1)
        should_check = ~ignore_matrix[idx_i, idx_j]
        active_i = idx_i[should_check]
        active_j = idx_j[should_check]

        return active_i, active_j

    @staticmethod
    def _get_trimesh_collision_geometries(
        urdf: yourdfpy.URDF, link_name: str
    ) -> trimesh.Trimesh:
        """Extracts trimesh collision geometries for a given link name, applying relative transforms."""
        if link_name not in urdf.link_map:
            return trimesh.Trimesh()

        link = urdf.link_map[link_name]
        filename_handler = urdf._filename_handler
        coll_meshes = []

        for collision in link.collisions:
            geom = collision.geometry
            mesh: Optional[trimesh.Trimesh] = None

            # Get the transform of the collision geometry relative to the link frame
            if collision.origin is not None:
                transform = collision.origin
            else:
                transform = jaxlie.SE3.identity().as_matrix()

            if geom.box is not None:
                mesh = trimesh.creation.box(extents=geom.box.size)
            elif geom.cylinder is not None:
                mesh = trimesh.creation.cylinder(
                    radius=geom.cylinder.radius, height=geom.cylinder.length
                )
            elif geom.sphere is not None:
                mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
            elif geom.mesh is not None:
                try:
                    mesh_path = geom.mesh.filename
                    loaded_obj = trimesh.load(
                        file_obj=filename_handler(mesh_path), force="mesh"
                    )

                    scale = (
                        geom.mesh.scale
                        if geom.mesh.scale is not None
                        else [1.0, 1.0, 1.0]
                    )

                    if isinstance(loaded_obj, trimesh.Trimesh):
                        mesh = loaded_obj.copy()
                        mesh.apply_scale(scale)
                    elif isinstance(loaded_obj, trimesh.Scene):
                        if len(loaded_obj.geometry) > 0:
                            geom_candidate = list(loaded_obj.geometry.values())[0]
                            if isinstance(geom_candidate, trimesh.Trimesh):
                                mesh = geom_candidate.copy()
                                mesh.apply_scale(scale)
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue  # Skip if load result is unexpected

                    if mesh:
                        mesh.fix_normals()

                except Exception as e:
                    logger.error(
                        f"Failed processing mesh '{geom.mesh.filename}' for link '{link_name}': {e}"
                    )
                    continue
            else:
                logger.warning(
                    f"Unsupported collision geometry type for link '{link_name}'."
                )
                continue

            if mesh is not None:
                # Apply the transform specified in the URDF collision tag
                mesh.apply_transform(transform)
                coll_meshes.append(mesh)

        coll_mesh = sum(coll_meshes, trimesh.Trimesh())
        return coll_mesh

    @jdc.jit
    def at_config(
        self, robot: Robot, cfg: Float[Array, "*batch actuated_count"]
    ) -> CollGeom:
        """
        Returns the collision geometry transformed to the given robot configuration.

        Ensures that the link transforms returned by forward kinematics are applied
        to the corresponding collision geometries stored in this object, based on link names.

        Args:
            robot: The Robot instance containing kinematics information.
            cfg: The robot configuration (actuated joints).

        Returns:
            The collision geometry (CollGeom) transformed to the world frame
            according to the provided configuration.
        """
        # Check if the link names match - this should be true if both Robot
        # and RobotCollision were created from the same URDF parser results.
        assert self.link_names == robot.links.names, (
            "Link name mismatch between RobotCollision and Robot kinematics."
        )

        Ts_link_world_wxyz_xyz = robot.forward_kinematics(cfg)
        Ts_link_world = jaxlie.SE3(Ts_link_world_wxyz_xyz)

        return self.coll.transform(Ts_link_world)

    def get_swept_capsules(
        self,
        robot: Robot,
        cfg_prev: Float[Array, "*batch actuated_count"],
        cfg_next: Float[Array, "*batch actuated_count"],
    ) -> Capsule:
        """
        Computes swept-volume capsules between two configurations.

        For each link, the capsule at cfg_prev and cfg_next is decomposed into
        a fixed number of spheres (currently 5). Corresponding sphere pairs are
        then connected by capsules to represent the swept volume.

        Args:
            robot: The Robot instance.
            cfg_prev: The starting robot configuration.
            cfg_next: The ending robot configuration.

        Returns:
            A Capsule object representing the swept volumes.
            The batch axes will be (*batch, 5, num_links).
        """
        n_segments = 5

        # 1. Get collision geometries at start and end configurations
        # Shape: (*batch, num_links)
        coll_prev_world: Capsule = cast(Capsule, self.at_config(robot, cfg_prev))
        coll_next_world: Capsule = cast(Capsule, self.at_config(robot, cfg_next))
        assert isinstance(coll_prev_world, Capsule)
        assert isinstance(coll_next_world, Capsule)
        assert coll_prev_world.get_batch_axes() == coll_next_world.get_batch_axes()

        # 2. Decompose capsules into spheres
        # Shape: (n_segments, *batch, num_links)
        spheres_prev = coll_prev_world.decompose_to_spheres(n_segments)
        spheres_next = coll_next_world.decompose_to_spheres(n_segments)
        assert spheres_prev.get_batch_axes() == spheres_next.get_batch_axes(), (
            "Sphere batch axes mismatch after decomposition."
        )
        expected_sphere_batch_axes = (
            (n_segments,) + cfg_prev.shape[:-1] + (self.num_links,)
        )
        assert spheres_prev.get_batch_axes() == expected_sphere_batch_axes, (
            f"Unexpected sphere batch axes: {spheres_prev.get_batch_axes()} vs {expected_sphere_batch_axes}"
        )

        # 3. Create swept capsules by connecting corresponding sphere pairs
        # Shape: (n_segments, *batch, num_links)
        swept_capsules = Capsule.from_sphere_pairs(spheres_prev, spheres_next)
        assert swept_capsules.get_batch_axes() == expected_sphere_batch_axes, (
            "Swept capsule batch axes mismatch."
        )

        # The result contains capsules for each segment of each link.
        return swept_capsules

    def compute_self_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_active_pairs"]:
        """
        Computes the signed distances for active self-collision pairs.

        Args:
            robot_coll: The robot's collision model with precomputed active pair indices.
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).

        Returns:
            Signed distances for each active pair.
            Shape: (*batch, num_active_pairs).
            Positive distance means separation, negative means penetration.
        """
        batch_axes = cfg.shape[:-1]

        # 1. Get collision geometry at the current config
        coll = self.at_config(robot, cfg)
        assert coll.get_batch_axes() == (*batch_axes, self.num_links)

        # 2. Compute all pairwise distances using the imported function
        dist_matrix = pairwise_collide(coll, coll)
        assert dist_matrix.shape == (
            *batch_axes,
            self.num_links,
            self.num_links,
        )

        # 3. Extract distances for the precomputed active pairs
        # Use advanced indexing with the stored indices
        active_distances = dist_matrix[..., self.active_idx_i, self.active_idx_j]

        # Expected shape check
        num_active_pairs = len(self.active_idx_i)
        assert active_distances.shape == (*batch_axes, num_active_pairs)

        return active_distances

    def compute_world_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: CollGeom,  # Shape: (*batch_world, M, ...)
    ) -> Float[Array, "*batch_combined N M"]:
        """
        Computes the signed distances between all robot links (N) and all world obstacles (M).

        Args:
            robot_coll: The robot's collision model.
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).
            world_geom: Collision geometry representing world obstacles. If representing a
                single obstacle, it should have batch shape (). If multiple, the last axis
                is interpreted as the collection of world objects (M).
                The batch dimensions (*batch_world) must be broadcast-compatible with cfg's
                batch axes (*batch_cfg).

        Returns:
            Matrix of signed distances between each robot link and each world object.
            Shape: (*batch_combined, N, M), where N=num_links, M=num_world_objects.
            Positive distance means separation, negative means penetration.
        """
        # 1. Get robot collision geometry at the current config
        # Shape: (*batch_cfg, N, ...)
        coll_robot_world = self.at_config(robot, cfg)
        N = self.num_links
        assert coll_robot_world.get_batch_axes()[-1] == N
        batch_cfg_shape = coll_robot_world.get_batch_axes()[:-1]

        # 2. Normalize world_geom shape and determine M
        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:  # Single world object
            # Use the object's broadcast_to method to add the M=1 axis correctly
            _world_geom = world_geom.broadcast_to((1,))
            M = 1
            batch_world_shape = ()
        else:  # Multiple world objects
            _world_geom = world_geom
            M = world_axes[-1]
            batch_world_shape = world_axes[:-1]

        # 3. Compute distances: Map collide over robot links (axis -2) vs _world_geom (None)
        # _world_geom is guaranteed to have the M axis now.
        _collide_links_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=(-2))
        dist_matrix = _collide_links_vs_world(coll_robot_world, _world_geom)

        # 4. Result shape check
        # Calculate expected shape based on broadcasting rules
        expected_batch_combined = jnp.broadcast_shapes(
            batch_cfg_shape, batch_world_shape
        )
        expected_shape = (*expected_batch_combined, N, M)

        # Perform the assertion without try-except or complex logic
        assert dist_matrix.shape == expected_shape, (
            f"Output shape mismatch. Expected {expected_shape}, Got {dist_matrix.shape}. "
            f"Robot axes: {coll_robot_world.get_batch_axes()}, Original World axes: {world_geom.get_batch_axes()}"
        )

        # 5. Return the distance matrix
        return dist_matrix
