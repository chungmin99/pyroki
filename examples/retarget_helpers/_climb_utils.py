"""Utilities for climbing retargeting.

This module provides functions for:
- Loading mocap motion data and object meshes
- Building interaction meshes via Delaunay triangulation
- Computing Laplacian coordinates for mesh deformation preservation
- Mapping between mocap joint names and G1 robot links
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as onp
import trimesh
from scipy.spatial import Delaunay

from ._utils import G1_LINK_NAMES

# =============================================================================
# Mocap Joint Names (from holosoma data_type.py:89-143)
# =============================================================================

MOCAP_DEMO_JOINTS = [
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftFootMod",
    "RightFootMod",
]

# Mapping from mocap joints to G1 robot links
# Based on holosoma ("mocap", "g1") mapping from data_type.py:215-231
# Uses link names from load_robot_description("g1_description")
# Format: "mocap_joint": ("g1_link", (x, y, z) local offset) or just "g1_link" for no offset
# Offsets derived from holosoma g1_29dof.urdf intermediate/sphere links
MOCAP_TO_G1_MAPPING: dict[str, str | tuple[str, tuple[float, float, float]]] = {
    "Spine1": "pelvis_contour_link",
    "LeftUpLeg": "left_hip_pitch_link",
    "LeftLeg": "left_knee_link",
    # LeftFoot: holosoma uses ankle_intermediate_1 which is 2cm higher than ankle_pitch
    "LeftFoot": ("left_ankle_pitch_link", (0.0, 0.0, 0.02)),
    # LeftToeBase: holosoma uses ankle_roll_sphere_5 at (0.14, 0, -0.03) from ankle_roll
    "LeftToeBase": ("left_ankle_roll_link", (0.14, 0.0, -0.03)),
    "RightUpLeg": "right_hip_pitch_link",
    "RightLeg": "right_knee_link",
    "RightFoot": ("right_ankle_pitch_link", (0.0, 0.0, 0.02)),
    "RightToeBase": ("right_ankle_roll_link", (0.14, 0.0, -0.03)),
    "LeftArm": "left_shoulder_roll_link",
    "LeftForeArm": "left_elbow_link",
    "LeftHand": "left_rubber_hand",
    "RightArm": "right_shoulder_roll_link",
    "RightForeArm": "right_elbow_link",
    "RightHand": "right_rubber_hand",
}


# =============================================================================
# Interaction Mesh Functions
# =============================================================================


def create_interaction_mesh(
    vertices: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray]:
    """Create a tetrahedral mesh from vertices using Delaunay triangulation.

    Args:
        vertices: (N, 3) array of vertex positions.

    Returns:
        Tuple of (vertices, tetrahedra) where tetrahedra is (M, 4) array of
        vertex indices forming the tetrahedra.
    """
    tri = Delaunay(vertices)
    return vertices, tri.simplices


def get_adjacency_list(
    tetrahedra: onp.ndarray, num_vertices: int
) -> list[list[int]]:
    """Build adjacency list from tetrahedra.

    Args:
        tetrahedra: (M, 4) array of vertex indices.
        num_vertices: Total number of vertices.

    Returns:
        List of lists, where adj[i] contains indices of vertices adjacent to i.
    """
    adj = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def calculate_laplacian_coordinates(
    vertices: jnp.ndarray,
    adj_list: list[list[int]],
    uniform_weight: bool = True,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """Calculate Laplacian coordinates for each vertex in the mesh.

    The Laplacian coordinate of vertex i is:
        L[i] = v[i] - weighted_center_of_neighbors

    This captures the local shape/structure around each vertex.

    Args:
        vertices: (N, 3) array of vertex positions.
        adj_list: Adjacency list from get_adjacency_list.
        uniform_weight: If True, use uniform weights (w=1).
                       If False, use distance-based weights (w=1/(1.5*dist+eps)).
        epsilon: Small value to prevent division by zero.

    Returns:
        (N, 3) array of Laplacian coordinates.
    """
    num_vertices = vertices.shape[0]
    laplacian = jnp.zeros_like(vertices)

    for i in range(num_vertices):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            vi = vertices[i]
            neighbor_positions = vertices[jnp.array(neighbors_indices)]

            if uniform_weight:
                weights = jnp.ones(len(neighbors_indices))
            else:
                distances = jnp.linalg.norm(vi - neighbor_positions, axis=1)
                weights = 1.0 / (1.5 * distances + epsilon)

            sum_of_weights = jnp.sum(weights)
            weighted_sum = jnp.sum(weights[:, None] * neighbor_positions, axis=0)
            center_of_neighbors = weighted_sum / sum_of_weights
            laplacian = laplacian.at[i].set(vi - center_of_neighbors)

    return laplacian


def calculate_laplacian_coordinates_vectorized(
    vertices: jnp.ndarray,
    adj_matrix: jnp.ndarray,
    uniform_weight: bool = True,
) -> jnp.ndarray:
    """Vectorized version of Laplacian coordinate calculation.

    Uses adjacency matrix instead of adjacency list for JAX compatibility.

    Args:
        vertices: (N, 3) array of vertex positions.
        adj_matrix: (N, N) binary adjacency matrix where adj[i,j]=1 if connected.
        uniform_weight: If True, use uniform weights.

    Returns:
        (N, 3) array of Laplacian coordinates.
    """
    # Compute degree (number of neighbors) for each vertex
    degree = jnp.sum(adj_matrix, axis=1, keepdims=True)  # (N, 1)
    degree = jnp.maximum(degree, 1.0)  # Avoid division by zero

    # Compute weighted sum of neighbors
    # adj_matrix: (N, N), vertices: (N, 3)
    neighbor_sum = adj_matrix @ vertices  # (N, 3)

    # Center of neighbors
    center_of_neighbors = neighbor_sum / degree

    # Laplacian = vertex - center_of_neighbors
    laplacian = vertices - center_of_neighbors

    return laplacian


def adjacency_list_to_matrix(
    adj_list: list[list[int]], num_vertices: int
) -> jnp.ndarray:
    """Convert adjacency list to adjacency matrix.

    Args:
        adj_list: List of lists of neighbor indices.
        num_vertices: Total number of vertices.

    Returns:
        (N, N) binary adjacency matrix.
    """
    adj_matrix = onp.zeros((num_vertices, num_vertices))
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            adj_matrix[i, j] = 1.0
    return jnp.array(adj_matrix)


# =============================================================================
# Data Loading Functions
# =============================================================================


def get_climb_retarget_indices(
    robot,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get mapping indices between mocap joints and G1 robot links.

    Args:
        robot: pyroki Robot instance.

    Returns:
        Tuple of (mocap_indices, g1_link_indices, local_offsets) where:
        - mocap_indices: indices into MOCAP_DEMO_JOINTS
        - g1_link_indices: indices into robot.links.names
        - local_offsets: (N, 3) array of local-frame offsets for each keypoint
    """
    mocap_indices = []
    g1_link_indices = []
    local_offsets = []

    link_names = robot.links.names

    for mocap_name, mapping_value in MOCAP_TO_G1_MAPPING.items():
        # Parse mapping: either "link_name" or ("link_name", (x, y, z))
        if isinstance(mapping_value, tuple):
            g1_name, offset = mapping_value
        else:
            g1_name = mapping_value
            offset = (0.0, 0.0, 0.0)

        if mocap_name in MOCAP_DEMO_JOINTS and g1_name in link_names:
            mocap_indices.append(MOCAP_DEMO_JOINTS.index(mocap_name))
            g1_link_indices.append(link_names.index(g1_name))
            local_offsets.append(offset)

    return jnp.array(mocap_indices), jnp.array(g1_link_indices), jnp.array(local_offsets)


def load_climb_motion(
    path: Path, downsample: int = 4, scale_factor: float = 0.742
) -> jnp.ndarray:
    """Load mocap motion data for climbing.

    Args:
        path: Path to the .npy file containing joint positions.
        downsample: Downsample factor (holosoma uses 4x).
        scale_factor: Scale factor for robot height (default 0.714 for G1).

    Returns:
        (T, num_joints, 3) array of joint positions.
    """
    motion = onp.load(path)  # Shape: (T, num_joints, 3)

    # Downsample
    motion = motion[::downsample]

    # Apply scale factor
    motion = motion * scale_factor

    return jnp.array(motion)


def load_object_points(
    obj_path: Path,
    sample_count: int = 50,
    seed: int = 42,
    scale_factor: float = 0.742,
) -> jnp.ndarray:
    """Load and sample surface points from object mesh.

    Args:
        obj_path: Path to the .obj mesh file.
        sample_count: Number of points to sample from surface.
        seed: Random seed for sampling.
        scale_factor: Scale factor to match robot scale.

    Returns:
        (sample_count, 3) array of sampled surface points.
    """
    mesh = trimesh.load(obj_path, force="mesh")
    points, _ = trimesh.sample.sample_surface_even(mesh, sample_count, seed=seed)
    points = onp.array(points) * scale_factor
    return jnp.array(points)


def load_object_mesh(obj_path: Path, scale_factor: float = 0.742) -> trimesh.Trimesh:
    """Load object mesh for visualization.

    Args:
        obj_path: Path to the .obj mesh file.
        scale_factor: Scale factor to match robot scale.

    Returns:
        Scaled trimesh object.
    """
    mesh = trimesh.load(obj_path, force="mesh")
    mesh.apply_scale(scale_factor)
    return mesh


# =============================================================================
# Foot Contact Detection (ported from holosoma)
# =============================================================================


def extract_foot_sticking_sequence_velocity(
    keypoints: onp.ndarray,
    joint_names: list[str],
    foot_names: tuple[str, str] = ("LeftToeBase", "RightToeBase"),
    velocity_threshold: float = 0.01,
) -> onp.ndarray:
    """Extract foot contact sequence based on XY velocity of foot joints.

    A foot is considered in contact when its XY velocity is below the threshold.
    This is ported from holosoma's utils.py.

    Args:
        keypoints: (T, num_joints, 3) array of joint positions.
        joint_names: List of joint names corresponding to keypoints axis 1.
        foot_names: Tuple of (left_foot_name, right_foot_name) joint names.
        velocity_threshold: Threshold for XY velocity to determine contact.
            Default 0.01 (10mm per frame).

    Returns:
        (T, 2) boolean array where [:, 0] is left foot contact, [:, 1] is right foot.
    """
    left_idx = joint_names.index(foot_names[0])
    right_idx = joint_names.index(foot_names[1])

    # XY positions only
    left_pos = keypoints[:, left_idx, :2]
    right_pos = keypoints[:, right_idx, :2]

    # Compute XY velocities (frame-to-frame displacement)
    left_vel = onp.linalg.norm(onp.diff(left_pos, axis=0), axis=1)
    right_vel = onp.linalg.norm(onp.diff(right_pos, axis=0), axis=1)

    # Pad first frame (set to not in contact since we have no velocity info)
    left_vel = onp.concatenate([[velocity_threshold + 1], left_vel])
    right_vel = onp.concatenate([[velocity_threshold + 1], right_vel])

    # Contact when velocity below threshold
    left_contact = left_vel <= velocity_threshold
    right_contact = right_vel <= velocity_threshold

    return onp.stack([left_contact, right_contact], axis=1)
