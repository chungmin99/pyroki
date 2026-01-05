from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import trimesh
import yourdfpy
from jaxtyping import Array, Float, Int
from loguru import logger

if TYPE_CHECKING:
    from pyroki._robot import Robot

from .._robot_urdf_parser import RobotURDFParser
from ._collision import collide, pairwise_collide
from ._geometry import Capsule, CollGeom, Sphere


@jdc.pytree_dataclass
class RobotCollision:
    """Collision model for a robot, integrated with pyroki kinematics."""

    num_links: jdc.Static[int]
    """Number of links in the model (matches kinematics links)."""
    link_names: jdc.Static[tuple[str, ...]]
    """Names of the links corresponding to link indices."""
    coll: CollGeom
    """Collision geometries for the robot (relative to their parent link frame)."""

    active_idx_i: jdc.Static[tuple[int, ...]]
    """Row indices (first link) of active self-collision pairs to check."""
    active_idx_j: jdc.Static[tuple[int, ...]]
    """Column indices (second link) of active self-collision pairs to check."""

    # Fields for sphere-based collision (variable geometries per link)
    max_geoms_per_link: jdc.Static[int] = 1
    """Maximum number of geometries (spheres) per link. Used for padding."""
    geom_counts: Int[Array, " num_links"] | None = None
    """Actual number of geometries per link (None for capsule mode)."""

    # Geometry-pair indices for sphere self-collision
    geom_pair_link_i: jdc.Static[tuple[int, ...] | None] = None
    """Link index for first geometry in each active pair."""
    geom_pair_idx_i: jdc.Static[tuple[int, ...] | None] = None
    """Geometry index within link_i for each active pair."""
    geom_pair_link_j: jdc.Static[tuple[int, ...] | None] = None
    """Link index for second geometry in each active pair."""
    geom_pair_idx_j: jdc.Static[tuple[int, ...] | None] = None
    """Geometry index within link_j for each active pair."""

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
    def from_sphere_decomposition(
        sphere_decomposition: dict[str, dict],
        link_names: tuple[str, ...],
        urdf: yourdfpy.URDF | None = None,
        user_ignore_pairs: tuple[tuple[str, str], ...] = (),
        ignore_immediate_adjacents: bool = True,
    ) -> "RobotCollision":
        """
        Build a RobotCollision model from sphere decomposition data.

        Args:
            sphere_decomposition: Dict mapping link names to sphere specs.
                Format: {'link_name': {'centers': [[x,y,z], ...], 'radii': [r, ...]}, ...}
                Links not in this dict will have no collision geometry (empty).
            link_names: Ordered tuple of link names matching Robot.links.names.
            urdf: Optional URDF for computing ignore pairs from adjacency.
                Required if ignore_immediate_adjacents=True.
            user_ignore_pairs: Additional pairs of link names to ignore for self-collision.
            ignore_immediate_adjacents: If True, automatically ignore collisions
                between adjacent (parent/child) links based on the URDF structure.

        Returns:
            RobotCollision configured for sphere-based collision checking.
        """
        num_links = len(link_names)

        # Compute geometry counts and max_geoms_per_link
        geom_counts_list: list[int] = []
        for link_name in link_names:
            link_data = sphere_decomposition.get(link_name, {"centers": [], "radii": []})
            geom_counts_list.append(len(link_data.get("centers", [])))

        max_geoms = max(geom_counts_list) if geom_counts_list else 1
        max_geoms = max(max_geoms, 1)  # At least 1 to avoid zero-size arrays
        geom_counts = onp.array(geom_counts_list, dtype=onp.int32)

        # Build padded sphere arrays
        all_centers: list[list[list[float]]] = []
        all_radii: list[list[float]] = []

        for link_name in link_names:
            link_data = sphere_decomposition.get(link_name, {"centers": [], "radii": []})
            link_centers: list[list[float]] = []
            link_radii: list[float] = []

            for center, radius in zip(
                link_data.get("centers", []), link_data.get("radii", [])
            ):
                link_centers.append(list(center))
                link_radii.append(float(radius))

            # Pad with zero-radius spheres at origin
            while len(link_centers) < max_geoms:
                link_centers.append([0.0, 0.0, 0.0])
                link_radii.append(0.0)

            all_centers.append(link_centers)
            all_radii.append(link_radii)

        # Create batched Sphere with shape (num_links, max_geoms)
        centers_array = jnp.array(all_centers)  # (num_links, max_geoms, 3)
        radii_array = jnp.array(all_radii)  # (num_links, max_geoms)
        spheres = Sphere.from_center_and_radius(centers_array, radii_array)
        assert spheres.get_batch_axes() == (num_links, max_geoms)

        # Compute geometry-pair indices
        geom_pair_link_i, geom_pair_idx_i, geom_pair_link_j, geom_pair_idx_j = (
            RobotCollision._compute_geometry_pair_indices(
                link_names=link_names,
                geom_counts=geom_counts,
                urdf=urdf,
                user_ignore_pairs=user_ignore_pairs,
                ignore_immediate_adjacents=ignore_immediate_adjacents,
            )
        )

        # Also compute link-level active pairs for compatibility
        active_idx_i, active_idx_j = RobotCollision._compute_active_pair_indices_from_ignore_matrix(
            link_names=link_names,
            urdf=urdf,
            user_ignore_pairs=user_ignore_pairs,
            ignore_immediate_adjacents=ignore_immediate_adjacents,
        )

        logger.info(
            f"Created RobotCollision with {num_links} links, "
            f"{max_geoms} max spheres/link, and "
            f"{len(geom_pair_link_i)} active geometry pairs."
        )

        return RobotCollision(
            num_links=num_links,
            link_names=link_names,
            coll=spheres,
            active_idx_i=active_idx_i,
            active_idx_j=active_idx_j,
            max_geoms_per_link=max_geoms,
            geom_counts=jnp.array(geom_counts),
            geom_pair_link_i=geom_pair_link_i,
            geom_pair_idx_i=geom_pair_idx_i,
            geom_pair_link_j=geom_pair_link_j,
            geom_pair_idx_j=geom_pair_idx_j,
        )

    @staticmethod
    def _compute_active_pair_indices_from_ignore_matrix(
        link_names: tuple[str, ...],
        urdf: yourdfpy.URDF | None,
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Compute link-level active pair indices using the ignore matrix helper."""
        num_links = len(link_names)
        ignore_matrix = RobotCollision._build_ignore_matrix(
            link_names, urdf, user_ignore_pairs, ignore_immediate_adjacents
        )

        idx_i, idx_j = onp.tril_indices(num_links, k=-1)
        should_check = ~ignore_matrix[idx_i, idx_j]
        active_i = idx_i[should_check]
        active_j = idx_j[should_check]

        return (tuple(active_i.tolist()), tuple(active_j.tolist()))

    @staticmethod
    def _compute_active_pair_indices(
        link_names: tuple[str, ...],
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
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

        return (
            tuple(onp.array(active_i).tolist()),
            tuple(onp.array(active_j).tolist()),
        )

    @staticmethod
    def _build_ignore_matrix(
        link_names: tuple[str, ...],
        urdf: yourdfpy.URDF | None,
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
    ) -> onp.ndarray:
        """Build a boolean matrix indicating which link pairs should be ignored."""
        num_links = len(link_names)
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}
        ignore_matrix = onp.zeros((num_links, num_links), dtype=bool)

        # Ignore self-pairs
        onp.fill_diagonal(ignore_matrix, True)

        # Ignore adjacent links from URDF
        if ignore_immediate_adjacents and urdf is not None:
            for joint in urdf.joint_map.values():
                parent_name = joint.parent
                child_name = joint.child
                if parent_name in link_name_to_idx and child_name in link_name_to_idx:
                    parent_idx = link_name_to_idx[parent_name]
                    child_idx = link_name_to_idx[child_name]
                    ignore_matrix[parent_idx, child_idx] = True
                    ignore_matrix[child_idx, parent_idx] = True

        # Ignore user-specified pairs
        for name1, name2 in user_ignore_pairs:
            if name1 in link_name_to_idx and name2 in link_name_to_idx:
                idx1 = link_name_to_idx[name1]
                idx2 = link_name_to_idx[name2]
                ignore_matrix[idx1, idx2] = True
                ignore_matrix[idx2, idx1] = True

        return ignore_matrix

    @staticmethod
    def _compute_geometry_pair_indices(
        link_names: tuple[str, ...],
        geom_counts: onp.ndarray,
        urdf: yourdfpy.URDF | None,
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """
        Compute geometry-level pair indices for sphere self-collision.

        For each non-ignored link pair (i < j), enumerates all sphere pairs.

        Returns:
            Tuple of (link_i, idx_i, link_j, idx_j) where each is a tuple of indices.
        """
        num_links = len(link_names)
        ignore_matrix = RobotCollision._build_ignore_matrix(
            link_names, urdf, user_ignore_pairs, ignore_immediate_adjacents
        )

        pairs_link_i: list[int] = []
        pairs_idx_i: list[int] = []
        pairs_link_j: list[int] = []
        pairs_idx_j: list[int] = []

        for li in range(num_links):
            for lj in range(li + 1, num_links):
                if ignore_matrix[li, lj]:
                    continue
                # Enumerate all geometry pairs between these links
                for gi in range(geom_counts[li]):
                    for gj in range(geom_counts[lj]):
                        pairs_link_i.append(li)
                        pairs_idx_i.append(gi)
                        pairs_link_j.append(lj)
                        pairs_idx_j.append(gj)

        return (
            tuple(pairs_link_i),
            tuple(pairs_idx_i),
            tuple(pairs_link_j),
            tuple(pairs_idx_j),
        )

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

        if self.geom_counts is None:
            # Capsule path: coll shape (num_links,)
            return self.coll.transform(Ts_link_world)
        else:
            # Sphere path: coll shape (num_links, max_geoms)
            # FK shape: (*batch, num_links, 7) -> broadcast to (*batch, num_links, max_geoms, 7)
            Ts_expanded = jaxlie.SE3(
                jnp.broadcast_to(
                    Ts_link_world.wxyz_xyz[..., None, :],
                    Ts_link_world.wxyz_xyz.shape[:-1] + (self.max_geoms_per_link, 7),
                )
            )
            return self.coll.transform(Ts_expanded)

    def get_link_collision_meshes(self) -> dict[str, trimesh.Trimesh]:
        """Get collision meshes for each link in their local coordinate frames.

        Returns a dict mapping link_name -> trimesh in local link frame.
        The meshes are NOT transformed to world frame - useful for attaching
        to viser frames that are already positioned by ViserUrdf.update_cfg().
        """
        result: dict[str, trimesh.Trimesh] = {}
        for i, link_name in enumerate(self.link_names):
            if self.geom_counts is None:
                # Capsule mode: one geometry per link
                mesh = self.coll._create_one_mesh((i,))
            else:
                # Sphere mode: multiple geometries per link
                count = int(self.geom_counts[i])
                if count == 0:
                    mesh = trimesh.Trimesh()
                else:
                    meshes = [self.coll._create_one_mesh((i, j)) for j in range(count)]
                    mesh = cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))
            result[link_name] = mesh
        return result

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

        if self.geom_pair_link_i is None:
            # Capsule path: use pairwise_collide on all links
            assert coll.get_batch_axes() == (*batch_axes, self.num_links)

            # 2. Compute all pairwise distances using the imported function
            dist_matrix = pairwise_collide(coll, coll)
            assert dist_matrix.shape == (
                *batch_axes,
                self.num_links,
                self.num_links,
            )

            # 3. Extract distances for the precomputed active pairs
            idx_i = jnp.array(self.active_idx_i, dtype=jnp.int32)
            idx_j = jnp.array(self.active_idx_j, dtype=jnp.int32)
            active_distances = dist_matrix[..., idx_i, idx_j]

            num_active_pairs = len(self.active_idx_i)
            assert active_distances.shape == (*batch_axes, num_active_pairs)

            return active_distances
        else:
            # Sphere path: use geometry-pair indexing
            # coll shape: (*batch, num_links, max_geoms)
            assert coll.get_batch_axes() == (
                *batch_axes,
                self.num_links,
                self.max_geoms_per_link,
            )

            link_i = jnp.array(self.geom_pair_link_i, dtype=jnp.int32)
            idx_i = jnp.array(self.geom_pair_idx_i, dtype=jnp.int32)
            link_j = jnp.array(self.geom_pair_link_j, dtype=jnp.int32)
            idx_j = jnp.array(self.geom_pair_idx_j, dtype=jnp.int32)

            # Extract sphere pairs using advanced indexing
            # coll.pose.wxyz_xyz shape: (*batch, num_links, max_geoms, 7)
            # We need spheres at [link_i, idx_i] and [link_j, idx_j]
            spheres_i = jax.tree.map(lambda x: x[..., link_i, idx_i, :], coll)
            spheres_j = jax.tree.map(lambda x: x[..., link_j, idx_j, :], coll)

            # collide expects CollGeom objects, so reconstruct Spheres
            active_distances = collide(spheres_i, spheres_j)

            num_active_pairs = len(self.geom_pair_link_i)
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
        coll_robot_world = self.at_config(robot, cfg)
        N = self.num_links

        # 2. Normalize world_geom shape and determine M
        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:  # Single world object
            _world_geom = world_geom.broadcast_to((1,))
            M = 1
            batch_world_shape: tuple[int, ...] = ()
        else:  # Multiple world objects
            _world_geom = world_geom
            M = world_axes[-1]
            batch_world_shape = world_axes[:-1]

        if self.geom_counts is None:
            # Capsule path: coll shape (*batch_cfg, N)
            assert coll_robot_world.get_batch_axes()[-1] == N
            batch_cfg_shape = coll_robot_world.get_batch_axes()[:-1]

            # 3. Compute distances: Map collide over robot links (axis -2) vs _world_geom (None)
            _collide_links_vs_world = jax.vmap(
                collide, in_axes=(-2, None), out_axes=(-2)
            )
            dist_matrix = _collide_links_vs_world(coll_robot_world, _world_geom)

            # 4. Result shape check
            expected_batch_combined = jnp.broadcast_shapes(
                batch_cfg_shape, batch_world_shape
            )
            expected_shape = (*expected_batch_combined, N, M)

            assert dist_matrix.shape == expected_shape, (
                f"Output shape mismatch. Expected {expected_shape}, Got {dist_matrix.shape}. "
                f"Robot axes: {coll_robot_world.get_batch_axes()}, Original World axes: {world_geom.get_batch_axes()}"
            )

            return dist_matrix
        else:
            # Sphere path: coll shape (*batch_cfg, N, max_geoms)
            assert coll_robot_world.get_batch_axes()[-2:] == (
                N,
                self.max_geoms_per_link,
            )
            batch_cfg_shape = coll_robot_world.get_batch_axes()[:-2]

            # Compute distances for all spheres vs all world objects
            # vmap over links (axis -3) and geometries (axis -2)
            _collide_geoms_vs_world = jax.vmap(
                collide, in_axes=(-2, None), out_axes=(-2)
            )
            _collide_links_vs_world = jax.vmap(
                _collide_geoms_vs_world, in_axes=(-3, None), out_axes=(-3)
            )
            dist_full = _collide_links_vs_world(coll_robot_world, _world_geom)
            # dist_full shape: (*batch_combined, N, max_geoms, M)

            # Mask out padding spheres and take min over geometries per link
            # geom_counts shape: (N,)
            # valid_mask shape: (N, max_geoms)
            valid_mask = (
                jnp.arange(self.max_geoms_per_link)[None, :]
                < self.geom_counts[:, None]
            )
            # Broadcast mask to dist_full shape and apply
            # valid_mask needs shape: (N, max_geoms, 1) for broadcasting with (..., N, max_geoms, M)
            masked_dist = jnp.where(
                valid_mask[..., None], dist_full, jnp.inf
            )
            dist_matrix = jnp.min(masked_dist, axis=-2)  # (*batch_combined, N, M)

            # Result shape check
            expected_batch_combined = jnp.broadcast_shapes(
                batch_cfg_shape, batch_world_shape
            )
            expected_shape = (*expected_batch_combined, N, M)

            assert dist_matrix.shape == expected_shape, (
                f"Output shape mismatch. Expected {expected_shape}, Got {dist_matrix.shape}. "
                f"Robot axes: {coll_robot_world.get_batch_axes()}, Original World axes: {world_geom.get_batch_axes()}"
            )

            return dist_matrix
