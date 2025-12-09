from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float, Array

from ._geometry import HalfSpace, Sphere, Capsule, Box, Heightmap
from . import _utils


# --- HalfSpace Collision Implementations ---


def _halfspace_sphere_dist(
    halfspace_normal: Float[Array, "*batch 3"],
    halfspace_point: Float[Array, "*batch 3"],
    sphere_pos: Float[Array, "*batch 3"],
    sphere_radius: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between a halfspace boundary plane and sphere center, minus radius."""
    dist = (
        jnp.einsum("...i,...i->...", sphere_pos - halfspace_point, halfspace_normal)
        - sphere_radius
    )
    return dist


def halfspace_sphere(halfspace: HalfSpace, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculates distance between a halfspace and a sphere."""
    dist = _halfspace_sphere_dist(
        halfspace.normal,
        halfspace.pose.translation(),
        sphere.pose.translation(),
        sphere.radius,
    )
    return dist


def halfspace_capsule(halfspace: HalfSpace, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculates distance between halfspace and capsule (closest end)."""
    halfspace_normal = halfspace.normal
    halfspace_point = halfspace.pose.translation()
    cap_center = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.height[..., None] / 2
    dist1 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center + segment_offset, cap_radius
    )
    dist2 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center - segment_offset, cap_radius
    )
    final_dist = jnp.minimum(dist1, dist2)
    return final_dist


# --- Sphere/Capsule Collision Implementations ---


def _sphere_sphere_dist(
    pos1: Float[Array, "*batch 3"],
    radius1: Float[Array, "*batch"],
    pos2: Float[Array, "*batch 3"],
    radius2: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between two spheres."""
    _, dist_center = _utils.normalize_with_norm(pos2 - pos1)
    dist = dist_center - (radius1 + radius2)
    return dist


def sphere_sphere(sphere1: Sphere, sphere2: Sphere) -> Float[Array, "*batch"]:
    """Calculate distance between two spheres."""
    dist = _sphere_sphere_dist(
        sphere1.pose.translation(),
        sphere1.radius,
        sphere2.pose.translation(),
        sphere2.radius,
    )
    return dist


def sphere_capsule(sphere: Sphere, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between sphere and capsule."""
    cap_pos = capsule.pose.translation()
    sphere_pos = sphere.pose.translation()
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.height[..., None] / 2
    cap_a = cap_pos - segment_offset
    cap_b = cap_pos + segment_offset
    pt_on_axis = _utils.closest_segment_point(cap_a, cap_b, sphere_pos)
    dist = _sphere_sphere_dist(sphere_pos, sphere.radius, pt_on_axis, capsule.radius)
    return dist


def capsule_capsule(capsule1: Capsule, capsule2: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between two capsules."""
    pos1 = capsule1.pose.translation()
    axis1 = capsule1.axis
    length1 = capsule1.height
    radius1 = capsule1.radius
    segment1_offset = axis1 * length1[..., None] / 2
    a1 = pos1 - segment1_offset
    b1 = pos1 + segment1_offset

    pos2 = capsule2.pose.translation()
    axis2 = capsule2.axis
    length2 = capsule2.height
    radius2 = capsule2.radius
    segment2_offset = axis2 * length2[..., None] / 2
    a2 = pos2 - segment2_offset
    b2 = pos2 + segment2_offset

    pt1_on_axis, pt2_on_axis = _utils.closest_segment_to_segment_points(a1, b1, a2, b2)
    dist = _sphere_sphere_dist(pt1_on_axis, radius1, pt2_on_axis, radius2)
    return dist


# --- Heightmap Collision Implementations ---


def heightmap_sphere(heightmap: Heightmap, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and sphere.

    Approximation: Considers the heightmap point directly below the sphere center
    using bilinear interpolation and calculates vertical distance minus radius.
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), sphere.get_batch_axes()
    )

    sphere_pos_w = sphere.pose.translation()
    sphere_radius = sphere.radius
    interpolated_local_z = heightmap._interpolate_height_at_coords(sphere_pos_w)
    sphere_pos_h = heightmap.pose.inverse().apply(sphere_pos_w)
    sphere_local_z = sphere_pos_h[..., 2]
    dist = sphere_local_z - interpolated_local_z - sphere_radius

    assert dist.shape == batch_axes
    return dist


def heightmap_capsule(heightmap: Heightmap, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and capsule, by
    checking heightmap points below capsule endpoints.

    Note that this may miss collisions when capsule body intersects but endpoints are above heightmap!
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), capsule.get_batch_axes()
    )

    cap_pos_w = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis_w = capsule.axis  # World frame axis
    segment_offset_w = cap_axis_w * capsule.height[..., None] / 2

    # Calculate world positions of the two end-sphere centers.
    p1_w = cap_pos_w + segment_offset_w
    p2_w = cap_pos_w - segment_offset_w

    # Interpolate heightmap surface height (local Z) below each end-sphere center.
    h_surf1_local = heightmap._interpolate_height_at_coords(p1_w)
    h_surf2_local = heightmap._interpolate_height_at_coords(p2_w)

    # Get end-sphere centers Z coordinates in heightmap's local frame.
    p1_h = heightmap.pose.inverse().apply(p1_w)
    p2_h = heightmap.pose.inverse().apply(p2_w)
    z1_local = p1_h[..., 2]
    z2_local = p2_h[..., 2]

    # Calculate vertical distance for each end sphere.
    dist1 = z1_local - h_surf1_local - cap_radius
    dist2 = z2_local - h_surf2_local - cap_radius

    # Return the minimum distance.
    min_dist = jnp.minimum(dist1, dist2)
    assert min_dist.shape == batch_axes
    return min_dist


def heightmap_halfspace(
    heightmap: Heightmap, halfspace: HalfSpace
) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and halfspace.

    Approximation: Finds the minimum signed distance between any heightmap vertex
    and the halfspace plane.
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), halfspace.get_batch_axes()
    )

    # Heightmap vertices in world frame.
    verts_local = heightmap._get_vertices_local()  # (*batch, N, 3), N=H*W
    verts_world = heightmap.pose.apply(verts_local)  # (*batch, N, 3)

    # Halfspace plane properties (world frame).
    hs_normal_w = halfspace.normal  # (*batch, 3)
    hs_point_w = halfspace.pose.translation()  # (*batch, 3)

    # Ensure batch dimensions are compatible for broadcasting.
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), halfspace.get_batch_axes()
    )
    # Expand dims for broadcasting against vertices.
    hs_normal_w = jnp.broadcast_to(hs_normal_w, batch_axes + (3,))[..., None, :]
    hs_point_w = jnp.broadcast_to(hs_point_w, batch_axes + (3,))[..., None, :]
    verts_world = jnp.broadcast_to(verts_world, batch_axes + verts_world.shape[-2:])

    # Calculate signed distance for each vertex to the plane:
    vertex_distances = jnp.einsum(
        "...vi,...i->...v", verts_world - hs_point_w, hs_normal_w.squeeze(-2)
    )

    # Find the minimum distance among all vertices.
    min_dist = jnp.min(vertex_distances, axis=-1)
    assert min_dist.shape == batch_axes
    return min_dist


# --- Box Collision Implementations ---


def halfspace_box(halfspace: HalfSpace, box: Box) -> Float[Array, "*batch"]:
    """Calculate distance between a halfspace and a box.

    Computes signed distance of all 8 corners to the halfspace plane,
    returns the minimum (most penetrating corner).
    """
    batch_axes = jnp.broadcast_shapes(halfspace.get_batch_axes(), box.get_batch_axes())

    # Get box corners in world frame: (*batch, 8, 3)
    corners = box.get_corners_world()

    # Halfspace plane properties
    hs_normal = halfspace.normal  # (*batch, 3)
    hs_point = halfspace.pose.translation()  # (*batch, 3)

    # Broadcast for corner computation
    hs_normal = jnp.broadcast_to(hs_normal, batch_axes + (3,))
    hs_point = jnp.broadcast_to(hs_point, batch_axes + (3,))
    corners = jnp.broadcast_to(corners, batch_axes + (8, 3))

    # Compute signed distance for each corner: dot(corner - point, normal)
    # corners: (*batch, 8, 3), hs_point: (*batch, 3), hs_normal: (*batch, 3)
    corner_dists = jnp.einsum(
        "...ci,...i->...c", corners - hs_point[..., None, :], hs_normal
    )  # (*batch, 8)

    # Return minimum distance across all corners
    min_dist = jnp.min(corner_dists, axis=-1)
    return min_dist


def _sphere_box_dist(
    sphere_pos: Float[Array, "*batch 3"],
    sphere_radius: Float[Array, "*batch"],
    box: Box,
) -> Float[Array, "*batch"]:
    """Helper: Calculate distance between a sphere and a box."""
    # Transform sphere center to box's local frame
    local_center = box.pose.inverse().apply(sphere_pos)

    # Clamp to box bounds to find closest point on box surface
    half_ext = box.half_extent
    closest = jnp.clip(local_center, -half_ext, half_ext)

    # Distance from sphere center to closest point, minus radius
    diff = local_center - closest
    _, dist_to_surface = _utils.normalize_with_norm(diff)
    dist = dist_to_surface - sphere_radius

    return dist


def sphere_box(sphere: Sphere, box: Box) -> Float[Array, "*batch"]:
    """Calculate distance between a sphere and a box."""
    return _sphere_box_dist(
        sphere.pose.translation(),
        sphere.radius,
        box,
    )


def _point_to_box_sdf(
    point: Float[Array, "*batch 3"],
    half_ext: Float[Array, "*batch 3"],
) -> Float[Array, "*batch"]:
    """Signed distance from a point to an AABB centered at origin."""
    # Clamp point to box bounds
    closest = jnp.clip(point, -half_ext, half_ext)

    # Distance to closest point on box surface (with safe gradient via epsilon)
    diff = point - closest
    _, dist = _utils.normalize_with_norm(diff)

    # For inside case: compute penetration depth as distance to nearest face
    face_dists = half_ext - jnp.abs(point)  # Distance to each face (positive inside)
    min_face_dist = jnp.min(face_dists, axis=-1)

    # If inside (dist ~= 0), return negative penetration; otherwise return positive dist
    is_inside = dist < 1e-5
    return jnp.where(is_inside, -min_face_dist, dist)


def capsule_box(capsule: Capsule, box: Box) -> Float[Array, "*batch"]:
    """Calculate signed distance between a capsule and a box.

    Uses iterative closest point refinement between capsule axis and box,
    then computes proper signed distance using box SDF.
    """
    cap_pos = capsule.pose.translation()
    cap_axis = capsule.axis
    cap_radius = capsule.radius
    segment_offset = cap_axis * capsule.height[..., None] / 2

    # Capsule endpoints in world frame
    p1_world = cap_pos + segment_offset
    p2_world = cap_pos - segment_offset

    # Transform to box's local frame (box becomes AABB centered at origin)
    p1_local = box.pose.inverse().apply(p1_world)
    p2_local = box.pose.inverse().apply(p2_world)
    half_ext = box.half_extent

    # Iteration 1: closest point on segment to box center (origin)
    center = jnp.zeros_like(p1_local)
    pt_seg = _utils.closest_segment_point(p1_local, p2_local, center)
    pt_box = jnp.clip(pt_seg, -half_ext, half_ext)

    # Iteration 2: refine with closest point on segment to pt_box
    pt_seg = _utils.closest_segment_point(p1_local, p2_local, pt_box)

    # Compute signed distance from segment point to box
    sdf = _point_to_box_sdf(pt_seg, half_ext)

    return sdf - cap_radius
