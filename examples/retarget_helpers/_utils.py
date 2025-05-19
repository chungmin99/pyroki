import pyroki as pk
import jax.numpy as jnp


def create_conn_tree(robot: pk.Robot, link_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Create a NxN connectivity matrix for N links.
    The matrix is marked Y if there is a direct kinematic chain connection
    between the two links, without bypassing the root link.
    """
    n = len(link_indices)
    conn_matrix = jnp.zeros((n, n))

    def is_direct_chain_connection(idx1: int, idx2: int) -> bool:
        """Check if two joints are connected in the kinematic chain without other retargeted joints between"""
        joint1 = link_indices[idx1]
        joint2 = link_indices[idx2]

        # Check path from joint2 up to root
        current = joint2
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint1:
                return True
            if parent in link_indices:
                # Hit another retargeted joint before finding joint1
                break
            current = parent

        # Check path from joint1 up to root
        current = joint1
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint2:
                return True
            if parent in link_indices:
                # Hit another retargeted joint before finding joint2
                break
            current = parent

        return False

    # Build symmetric connectivity matrix
    for i in range(n):
        conn_matrix = conn_matrix.at[i, i].set(1.0)  # Self-connection
        for j in range(i + 1, n):
            if is_direct_chain_connection(i, j):
                conn_matrix = conn_matrix.at[i, j].set(1.0)
                conn_matrix = conn_matrix.at[j, i].set(1.0)

    return conn_matrix


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

# When loaded from `g1_description`s 23-dof model.
G1_LINK_NAMES = [
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_roll_link",
    "logo_link",
    "imu_link",
    "left_palm_link",
    "left_zero_link",
    "left_one_link",
    "left_two_link",
    "left_three_link",
    "left_four_link",
    "left_five_link",
    "left_six_link",
    "right_palm_link",
    "right_zero_link",
    "right_one_link",
    "right_two_link",
    "right_three_link",
    "right_four_link",
    "right_five_link",
    "right_six_link",
]


def get_humanoid_retarget_indices() -> tuple[jnp.ndarray, jnp.ndarray]:
    smpl_joint_retarget_indices_to_g1 = []
    g1_joint_retarget_indices = []

    for smpl_name, g1_name in [
        ("pelvis", "pelvis_contour_link"),
        ("left_hip", "left_hip_pitch_link"),
        ("right_hip", "right_hip_pitch_link"),
        ("left_knee", "left_knee_link"),
        ("right_knee", "right_knee_link"),
        ("left_ankle", "left_ankle_roll_link"),
        ("right_ankle", "right_ankle_roll_link"),
        ("left_shoulder", "left_shoulder_roll_link"),
        ("right_shoulder", "right_shoulder_roll_link"),
        ("left_elbow", "left_elbow_pitch_link"),
        ("right_elbow", "right_elbow_pitch_link"),
        ("left_wrist", "left_palm_link"),
        ("right_wrist", "right_palm_link"),
    ]:
        smpl_joint_retarget_indices_to_g1.append(SMPL_JOINT_NAMES.index(smpl_name))
        g1_joint_retarget_indices.append(G1_LINK_NAMES.index(g1_name))

    smpl_joint_retarget_indices = jnp.array(smpl_joint_retarget_indices_to_g1)
    g1_joint_retarget_indices = jnp.array(g1_joint_retarget_indices)
    return smpl_joint_retarget_indices, g1_joint_retarget_indices


MANO_TO_SHADOW_MAPPING = {
    # Wrist
    0: "palm",
    # Thumb
    1: "thhub",
    2: "thmiddle",
    3: "thdistal",
    4: "thtip",
    # Index
    5: "ffproximal",
    6: "ffmiddle",
    7: "ffdistal",
    8: "fftip",
    # Middle
    9: "mfproximal",
    10: "mfmiddle",
    11: "mfdistal",
    12: "mftip",
    # Ring
    13: "rfproximal",
    14: "rfmiddle",
    15: "rfdistal",
    16: "rftip",
    # # Little
    17: "lfproximal",
    18: "lfmiddle",
    19: "lfdistal",
    20: "lftip",
}


def get_mapping_from_mano_to_shadow(robot: pk.Robot) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the mapping indices between MANO and Shadow Hand joints."""
    SHADOW_TO_MANO_MAPPING = {v: k for k, v in MANO_TO_SHADOW_MAPPING.items()}
    shadow_joint_idx = []
    mano_joint_idx = []
    link_names = robot.links.names
    for i, link_name in enumerate(link_names):
        if link_name in SHADOW_TO_MANO_MAPPING:
            shadow_joint_idx.append(i)
            mano_joint_idx.append(SHADOW_TO_MANO_MAPPING[link_name])

    return jnp.array(shadow_joint_idx), jnp.array(mano_joint_idx)
