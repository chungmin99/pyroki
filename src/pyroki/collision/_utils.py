from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple
from jaxtyping import Float, Array

_SAFE_EPS = 1e-6


def make_frame(direction: jax.Array) -> jax.Array:
    """Make a frame from a direction vector, aligning the z-axis with the direction."""
    # Based on `mujoco.mjx._src.math.make_frame`.

    is_zero = jnp.isclose(direction, 0.0).all(axis=-1, keepdims=True)
    direction = jnp.where(
        is_zero,
        jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), direction.shape),
        direction,
    )
    direction /= jnp.linalg.norm(direction, axis=-1, keepdims=True) + _SAFE_EPS

    y = jnp.broadcast_to(jnp.array([0, 1, 0]), (*direction.shape[:-1], 3))
    z = jnp.broadcast_to(jnp.array([0, 0, 1]), (*direction.shape[:-1], 3))

    normal = jnp.where((-0.5 < direction[..., 1:2]) & (direction[..., 1:2] < 0.5), y, z)
    normal -= direction * jnp.einsum("...i,...i->...", normal, direction)[..., None]
    normal /= jnp.linalg.norm(normal, axis=-1, keepdims=True) + _SAFE_EPS

    return jnp.stack([jnp.cross(normal, direction), normal, direction], axis=-1)


def normalize(x: Float[Array, "*batch N"]) -> Float[Array, "*batch N"]:
    """Normalizes a vector, handling the zero vector."""
    norm: jax.Array = jnp.linalg.norm(x, axis=-1, keepdims=True)
    safe_norm = jnp.where(norm == 0.0, 1.0, norm)
    normalized_x = x / safe_norm
    return jnp.where(norm == 0.0, jnp.zeros_like(x), normalized_x)


def normalize_with_norm(
    x: Float[Array, "*batch N"],
) -> Tuple[Float[Array, "*batch N"], Float[Array, "*batch"]]:
    """Normalizes a vector and returns the norm, handling the zero vector."""
    norm: jax.Array = jnp.linalg.norm(x + 1e-6, axis=-1, keepdims=True)
    safe_norm = jnp.where(norm == 0.0, 1.0, norm)
    normalized_x = x / safe_norm
    result_vec = jnp.where(norm == 0.0, jnp.zeros_like(x), normalized_x)
    result_norm = norm[..., 0]
    return result_vec, result_norm


def closest_segment_point(
    a: Float[Array, "*batch 3"],
    b: Float[Array, "*batch 3"],
    pt: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 3"]:
    """Finds the closest point on the line segment [a, b] to point pt."""
    ab = b - a
    t = jnp.einsum("...i,...i->...", pt - a, ab) / (
        jnp.einsum("...i,...i->...", ab, ab) + _SAFE_EPS
    )
    t_clamped = jnp.clip(t, 0.0, 1.0)
    return a + ab * t_clamped[..., None]


def closest_segment_to_segment_points(
    a1: Float[Array, "*batch 3"],
    b1: Float[Array, "*batch 3"],
    a2: Float[Array, "*batch 3"],
    b2: Float[Array, "*batch 3"],
) -> Tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
    """Finds the closest points between two line segments [a1, b1] and [a2, b2]."""
    d1 = b1 - a1  # Direction vector of segment S1
    d2 = b2 - a2  # Direction vector of segment S2
    r = a1 - a2

    a = jnp.einsum("...i,...i->...", d1, d1)  # Squared length of segment S1
    e = jnp.einsum("...i,...i->...", d2, d2)  # Squared length of segment S2
    f = jnp.einsum("...i,...i->...", d2, r)
    c = jnp.einsum("...i,...i->...", d1, r)
    b = jnp.einsum("...i,...i->...", d1, d2)
    denom = a * e - b * b  # Squared area of the parallelogram defined by d1, d2

    s_num = b * f - c * e
    t_num = a * f - b * c

    s_parallel = -c / (a + _SAFE_EPS)
    t_parallel = f / (e + _SAFE_EPS)

    s = jnp.where(denom < _SAFE_EPS, s_parallel, s_num / (denom + _SAFE_EPS))
    t = jnp.where(denom < _SAFE_EPS, t_parallel, t_num / (denom + _SAFE_EPS))

    s_clamped = jnp.clip(s, 0.0, 1.0)
    t_clamped = jnp.clip(t, 0.0, 1.0)

    t_recomp = jnp.einsum(
        "...i,...i->...", d2, (a1 + d1 * s_clamped[..., None]) - a2
    ) / (e + _SAFE_EPS)
    t_final = jnp.where(
        jnp.abs(s - s_clamped) > _SAFE_EPS, jnp.clip(t_recomp, 0.0, 1.0), t_clamped
    )

    s_recomp = jnp.einsum("...i,...i->...", d1, (a2 + d2 * t_final[..., None]) - a1) / (
        a + _SAFE_EPS
    )
    s_final = jnp.where(
        jnp.abs(t - t_final) > _SAFE_EPS, jnp.clip(s_recomp, 0.0, 1.0), s_clamped
    )

    c1 = a1 + d1 * s_final[..., None]
    c2 = a2 + d2 * t_final[..., None]
    return c1, c2
