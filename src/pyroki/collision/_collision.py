from __future__ import annotations

from typing import Callable, Dict, Tuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._geometry import Capsule, CollGeom, HalfSpace, Heightmap, Sphere
from ._geometry_pairs import (
    capsule_capsule,
    halfspace_capsule,
    halfspace_sphere,
    heightmap_capsule,
    heightmap_halfspace,
    heightmap_sphere,
    sphere_capsule,
    sphere_sphere,
)

COLLISION_FUNCTIONS: Dict[
    Tuple[type[CollGeom], type[CollGeom]], Callable[..., Float[Array, "*batch"]]
] = {
    (HalfSpace, Sphere): halfspace_sphere,
    (HalfSpace, Capsule): halfspace_capsule,
    (Sphere, Sphere): sphere_sphere,
    (Sphere, Capsule): sphere_capsule,
    (Capsule, Capsule): capsule_capsule,
    (Heightmap, Sphere): heightmap_sphere,
    (Heightmap, Capsule): heightmap_capsule,
    (Heightmap, HalfSpace): heightmap_halfspace,
}


def _get_coll_func(
    geom1_cls: type[CollGeom], geom2_cls: type[CollGeom]
) -> Callable[[CollGeom, CollGeom], Float[Array, "*batch"]]:
    """Get appropriate collision function (distance only) for given geometry types."""
    func = COLLISION_FUNCTIONS.get((geom1_cls, geom2_cls))
    if func is not None:
        return cast(Callable[[CollGeom, CollGeom], Float[Array, "*batch"]], func)

    func_swapped = COLLISION_FUNCTIONS.get((geom2_cls, geom1_cls))
    if func_swapped is not None:
        return cast(
            Callable[[CollGeom, CollGeom], Float[Array, "*batch"]],
            lambda g1, g2: func_swapped(g2, g1),
        )

    raise NotImplementedError(
        f"No collision function found for {geom1_cls.__name__} and {geom2_cls.__name__}"
    )


def collide(geom1: CollGeom, geom2: CollGeom) -> Float[Array, "*batch"]:
    """Calculate collision distance between two geometric objects, handling broadcasting."""
    try:
        broadcast_shape = jnp.broadcast_shapes(
            geom1.get_batch_axes(), geom2.get_batch_axes()
        )
    except ValueError as e:
        raise ValueError(
            f"Cannot broadcast geometry shapes {geom1.get_batch_axes()} and {geom2.get_batch_axes()}"
        ) from e

    geom1 = geom1.broadcast_to(broadcast_shape)
    geom2 = geom2.broadcast_to(broadcast_shape)
    geom1_cls = type(geom1)
    geom2_cls = type(geom2)
    return _get_coll_func(geom1_cls, geom2_cls)(geom1, geom2)


def pairwise_collide(geom1: CollGeom, geom2: CollGeom) -> Float[Array, "*batch N M"]:
    """
    Convenience wrapper around `collide` for computing pairwise distances with broadcasting.

    Args:
        geom1: First collection of geometries. Expected to have a shape like
               (*batch1, N, ...), where N is the number of geometries.
        geom2: Second collection of geometries. Expected to have a shape like
               (*batch2, M, ...), where M is the number of geometries.
               The batch dimensions (*batch1, *batch2) must be broadcast-compatible.

    Returns:
        A matrix of distances with shape (*batch_combined, N, M), where
        *batch_combined is the result of broadcasting *batch1 and *batch2.
        dist[..., i, j] is the distance between geom1[..., i, :] and geom2[..., j, :].
    """
    # Input checks.
    axes1 = geom1.get_batch_axes()
    axes2 = geom2.get_batch_axes()
    assert len(axes1) >= 1, (
        f"geom1 must have at least one batch dimension to map over, got shape {axes1}"
    )
    assert len(axes2) >= 1, (
        f"geom2 must have at least one batch dimension to map over, got shape {axes2}"
    )

    # Determine expected output shape.
    batch1_shape = axes1[:-1]
    batch2_shape = axes2[:-1]
    N = axes1[-1]
    M = axes2[-1]
    try:
        batch_combined_shape = jnp.broadcast_shapes(batch1_shape, batch2_shape)
    except ValueError as e:
        raise ValueError(
            f"Cannot broadcast non-mapped batch shapes {batch1_shape} and {batch2_shape}"
        ) from e
    expected_output_shape = (*batch_combined_shape, N, M)
    result = collide(
        geom1.broadcast_to((*batch_combined_shape, N))
        .reshape((*batch_combined_shape, N, 1))
        .broadcast_to(expected_output_shape),
        geom2.broadcast_to((*batch_combined_shape, M))
        .reshape((*batch_combined_shape, 1, M))
        .broadcast_to(expected_output_shape),
    )
    assert result.shape == expected_output_shape, (
        f"Output shape mismatch. Expected {expected_output_shape}, got {result.shape}"
    )
    return result


def colldist_from_sdf(
    _dist: jax.Array,
    activation_dist: jax.Array | float,
) -> jax.Array:
    """
    Convert a signed distance field to a collision distance field,
    based on https://arxiv.org/pdf/2310.17274#page=7.39.

    This function applies a smoothing transformation, useful for converting
    raw distances into values suitable for cost functions in optimization.
    It returns values <= 0, where 0 corresponds to distances >= activation_dist,
    and increasingly negative values for deeper penetrations.

    Args:
        _dist: Signed distance field values (positive = separation, negative = penetration).
        activation_dist: The distance threshold (margin) below which the cost activates.

    Returns:
        Transformed collision distance field values (<= 0).
    """
    _dist = jnp.minimum(_dist, activation_dist)
    _dist = jnp.where(
        _dist < 0,
        _dist - 0.5 * activation_dist,
        -0.5 / (activation_dist + 1e-6) * (_dist - activation_dist) ** 2,
    )
    _dist = jnp.minimum(_dist, 0.0)
    return _dist
