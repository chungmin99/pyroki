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
from ._collision import collide
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
    """First index of active self-collision pairs (link indices for capsule, geometry indices for sphere)."""
    active_idx_j: jdc.Static[tuple[int, ...]]
    """Second index of active self-collision pairs (link indices for capsule, geometry indices for sphere)."""

    _geom_to_link_idx: Int[Array, " num_geoms"]
    """Maps each geometry to its parent link index (for FK transform lookup)."""

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
        active_idx_i, active_idx_j = RobotCollision._compute_collision_pair_indices(
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
            _geom_to_link_idx=jnp.arange(link_info.num_links, dtype=jnp.int32),
        )

    @staticmethod
    def from_sphere_decomposition(
        sphere_decomposition: dict[str, dict[str, list]],
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...] = (),
        ignore_immediate_adjacents: bool = True,
    ) -> "RobotCollision":
        """
        Build a RobotCollision model from sphere decomposition data.

        Args:
            sphere_decomposition: Dict mapping link names to sphere specs.
                Format: {'link_name': {'centers': [[x,y,z], ...], 'radii': [r, ...]}, ...}
                Links not in this dict will have no collision geometry (empty).
            urdf: URDF object used to determine link names and compute ignore pairs.
            user_ignore_pairs: Additional pairs of link names to ignore for self-collision.
            ignore_immediate_adjacents: If True, automatically ignore collisions
                between adjacent (parent/child) links based on the URDF structure.

        Returns:
            RobotCollision configured for sphere-based collision checking.
        """
        _, link_info = RobotURDFParser.parse(urdf)
        link_names = link_info.names
        num_links = len(link_names)

        # Validate sphere_decomposition structure
        assert all(
            d.keys() == {"centers", "radii"} for d in sphere_decomposition.values()
        )
        for link_name, link_data in sphere_decomposition.items():
            assert isinstance(link_data, dict)
            centers = link_data.get("centers", [])
            radii = link_data.get("radii", [])
            assert centers is not None and radii is not None
            assert len(centers) == len(radii)
            assert all(len(c) == 3 for c in centers)
            assert all(r >= 0 for r in radii)

        # Build flat sphere arrays and track link indices
        all_centers: list[list[float]] = []
        all_radii: list[float] = []
        sphere_link_indices: list[int] = []
        geom_counts: list[int] = []

        for link_idx, link_name in enumerate(link_names):
            link_data = sphere_decomposition.get(
                link_name, {"centers": [], "radii": []}
            )
            link_centers = link_data.get("centers", [])
            link_radii = link_data.get("radii", [])
            assert link_centers is not None and link_radii is not None

            geom_counts.append(len(link_centers))

            for center, radius in zip(link_centers, link_radii):
                all_centers.append(list(center))
                all_radii.append(float(radius))
                sphere_link_indices.append(link_idx)

        num_geoms = len(all_centers)
        assert num_geoms > 0, "No spheres found in the provided decomposition."

        # Create flat Sphere with shape (num_geoms,)
        centers_array = jnp.array(all_centers)  # (num_geoms, 3)
        radii_array = jnp.array(all_radii)  # (num_geoms,)
        spheres = Sphere.from_center_and_radius(centers_array, radii_array)
        assert spheres.get_batch_axes() == (num_geoms,)

        # Compute flat geometry-pair indices
        active_idx_i, active_idx_j = RobotCollision._compute_collision_pair_indices(
            link_names=link_names,
            urdf=urdf,
            user_ignore_pairs=user_ignore_pairs,
            ignore_immediate_adjacents=ignore_immediate_adjacents,
            geom_counts=onp.array(geom_counts, dtype=onp.int32),
        )

        logger.info(
            f"Created RobotCollision (sphere mode) with {num_links} links, "
            f"{num_geoms} total spheres, and {len(active_idx_i)} active pairs."
        )

        return RobotCollision(
            num_links=num_links,
            link_names=link_names,
            coll=spheres,
            active_idx_i=active_idx_i,
            active_idx_j=active_idx_j,
            _geom_to_link_idx=jnp.array(sphere_link_indices, dtype=jnp.int32),
        )

    @staticmethod
    def _compute_collision_pair_indices(
        link_names: tuple[str, ...],
        urdf: yourdfpy.URDF | None,
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
        geom_counts: onp.ndarray | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Compute collision pair indices for self-collision checking.

        When geom_counts is None (capsule mode), returns link-level indices.
        When geom_counts is provided (sphere mode), returns flat geometry-level indices.

        Args:
            link_names: Tuple of link names in order.
            urdf: URDF object for adjacency info.
            user_ignore_pairs: Pairs of link names to explicitly ignore.
            ignore_immediate_adjacents: Whether to ignore parent-child pairs from URDF.
            geom_counts: Array of geometry counts per link. If None, returns link indices.

        Returns:
            Tuple of (active_i, active_j) index arrays.
        """
        num_links = len(link_names)
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}

        # Build ignore set with normalized pairs (smaller index first).
        # Self-collision pairs not needed since loop uses li < lj.
        ignore_set: set[tuple[int, int]] = set()

        if ignore_immediate_adjacents and urdf is not None:
            for joint in urdf.joint_map.values():
                parent_name = joint.parent
                child_name = joint.child
                if parent_name in link_name_to_idx and child_name in link_name_to_idx:
                    parent_idx = link_name_to_idx[parent_name]
                    child_idx = link_name_to_idx[child_name]
                    ignore_set.add(
                        (min(parent_idx, child_idx), max(parent_idx, child_idx))
                    )

        for name1, name2 in user_ignore_pairs:
            if name1 in link_name_to_idx and name2 in link_name_to_idx:
                idx1 = link_name_to_idx[name1]
                idx2 = link_name_to_idx[name2]
                ignore_set.add((min(idx1, idx2), max(idx1, idx2)))

        # Treat capsule mode as 1 geometry per link.
        if geom_counts is None:
            geom_counts = onp.ones(num_links, dtype=onp.int32)

        geom_offsets = onp.zeros(num_links + 1, dtype=onp.int32)
        geom_offsets[1:] = onp.cumsum(geom_counts)

        idx_i: list[int] = []
        idx_j: list[int] = []

        for li in range(num_links):
            for lj in range(li + 1, num_links):
                if (li, lj) in ignore_set:
                    continue
                for gi in range(geom_counts[li]):
                    for gj in range(geom_counts[lj]):
                        idx_i.append(int(geom_offsets[li] + gi))
                        idx_j.append(int(geom_offsets[lj] + gj))

        return (tuple(idx_i), tuple(idx_j))

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

        # Index FK transforms by link for each geometry
        Ts_per_geom = jaxlie.SE3(Ts_link_world.wxyz_xyz[..., self._geom_to_link_idx, :])
        return self.coll.transform(Ts_per_geom)

    def get_link_collision_meshes(self) -> dict[str, trimesh.Trimesh]:
        """Get collision meshes for each link in their local coordinate frames.

        Returns a dict mapping link_name -> trimesh in local link frame.
        The meshes are NOT transformed to world frame - useful for attaching
        to viser frames that are already positioned by ViserUrdf.update_cfg().
        """
        result: dict[str, trimesh.Trimesh] = {}

        num_geoms = len(self._geom_to_link_idx)

        # Group geometry indices by link
        link_to_geom_indices: dict[int, list[int]] = {
            i: [] for i in range(self.num_links)
        }
        for geom_idx in range(num_geoms):
            link_idx = int(self._geom_to_link_idx[geom_idx])
            link_to_geom_indices[link_idx].append(geom_idx)

        for i, link_name in enumerate(self.link_names):
            geom_indices = link_to_geom_indices[i]
            if len(geom_indices) == 0:
                mesh = trimesh.Trimesh()
            else:
                meshes = [self.coll._create_one_mesh((j,)) for j in geom_indices]
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

        For Capsule mode: Each capsule is decomposed into 5 spheres along its
        axis. Corresponding sphere pairs are connected by capsules.
        Returns shape: (5, *batch, num_links)

        For Sphere mode: Each sphere's position at cfg_prev is connected to its
        position at cfg_next by a capsule. Returns shape: (*batch, num_geoms)

        Args:
            robot: The Robot instance.
            cfg_prev: The starting robot configuration.
            cfg_next: The ending robot configuration.

        Returns:
            A Capsule object representing the swept volumes.
        """
        coll_prev_world = self.at_config(robot, cfg_prev)
        coll_next_world = self.at_config(robot, cfg_next)
        assert coll_prev_world.get_batch_axes() == coll_next_world.get_batch_axes()

        if isinstance(coll_prev_world, Capsule):
            assert isinstance(coll_next_world, Capsule)
            n_segments = 5
            spheres_prev = coll_prev_world.decompose_to_spheres(n_segments)
            spheres_next = coll_next_world.decompose_to_spheres(n_segments)
            assert spheres_prev.get_batch_axes() == spheres_next.get_batch_axes()
            expected_batch = (n_segments,) + cfg_prev.shape[:-1] + (self.num_links,)
            assert spheres_prev.get_batch_axes() == expected_batch
            swept_capsules = Capsule.from_sphere_pairs(spheres_prev, spheres_next)
            assert swept_capsules.get_batch_axes() == expected_batch
            return swept_capsules

        elif isinstance(coll_prev_world, Sphere):
            assert isinstance(coll_next_world, Sphere)
            return Capsule.from_sphere_pairs(coll_prev_world, coll_next_world)

        else:
            raise TypeError(
                f"Unsupported collision geometry type: {type(coll_prev_world)}"
            )

    def compute_self_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_active_pairs"]:
        """
        Computes the signed distances for active self-collision pairs.

        Args:
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).

        Returns:
            Signed distances for each active pair.
            Shape: (*batch, num_active_pairs).
            Positive distance means separation, negative means penetration.
        """
        batch_axes = cfg.shape[:-1]

        # Get collision geometry at the current config
        coll = self.at_config(robot, cfg)

        # Extract geometry pairs using precomputed indices
        idx_i = jnp.array(self.active_idx_i, dtype=jnp.int32)
        idx_j = jnp.array(self.active_idx_j, dtype=jnp.int32)

        coll_i = jax.tree.map(lambda x: x[..., idx_i, :], coll)
        coll_j = jax.tree.map(lambda x: x[..., idx_j, :], coll)

        active_distances = collide(coll_i, coll_j)

        num_active_pairs = len(self.active_idx_i)
        assert active_distances.shape == (*batch_axes, num_active_pairs)

        return active_distances

    def compute_world_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: CollGeom,  # Shape: (*batch_world, num_world, ...)
    ) -> Float[Array, "*batch_combined num_geoms num_world"]:
        """
        Computes the signed distances between all robot geometries and all world obstacles.

        Args:
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).
            world_geom: Collision geometry representing world obstacles. If representing a
                single obstacle, it should have batch shape (). If multiple, the last axis
                is interpreted as the collection of world objects (num_world).
                The batch dimensions (*batch_world) must be broadcast-compatible with cfg's
                batch axes (*batch_cfg).

        Returns:
            Matrix of signed distances between each robot geometry and each world object.
            Shape: (*batch_combined, num_geoms, num_world), where num_geoms is the number of
            collision geometries (num_links for capsule mode, total spheres for sphere mode).
            Positive distance means separation, negative means penetration.
        """
        # Get robot collision geometry at the current config
        coll_robot_world = self.at_config(robot, cfg)

        # Derive num_geoms from collision geometry batch axes
        num_geoms = coll_robot_world.get_batch_axes()[-1]
        batch_cfg_shape = coll_robot_world.get_batch_axes()[:-1]

        # Normalize world_geom shape and determine num_world
        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:  # Single world object
            _world_geom = world_geom.broadcast_to((1,))
            num_world = 1
            batch_world_shape: tuple[int, ...] = ()
        else:  # Multiple world objects
            _world_geom = world_geom
            num_world = world_axes[-1]
            batch_world_shape = world_axes[:-1]

        # Compute distances: vmap collide over robot geometries vs world objects
        _collide_geoms_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=(-2))
        dist_matrix = _collide_geoms_vs_world(coll_robot_world, _world_geom)

        # Result shape check
        expected_batch_combined = jnp.broadcast_shapes(
            batch_cfg_shape, batch_world_shape
        )
        expected_shape = (*expected_batch_combined, num_geoms, num_world)

        assert dist_matrix.shape == expected_shape, (
            f"Output shape mismatch. Expected {expected_shape}, Got {dist_matrix.shape}. "
            f"Robot axes: {coll_robot_world.get_batch_axes()}, Original World axes: {world_geom.get_batch_axes()}"
        )

        return dist_matrix
