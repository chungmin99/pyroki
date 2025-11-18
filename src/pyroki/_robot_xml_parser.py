"""MJCF parsing utilities for Robot class."""

import xml.etree.ElementTree as ET
from ._robot_urdf_parser import JointInfo, LinkInfo
from jax import Array
import jaxlie
from jax import numpy as jnp

def mj_quat_to_rotmat(q: list[float]) -> jnp.ndarray:
    """Convert MuJoCo quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q

    # (Assumes q is normalized; if not, normalize it)
    # n = np.linalg.norm(q)
    # w, x, y, z = q / n

    R = jnp.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R


def euler_xyz_to_rotmat(euler):
    """
    Convert XYZ Euler angles (roll, pitch, yaw) to a 3×3 rotation matrix.
    euler = (rx, ry, rz)
    Convention: R = Rz(rz) * Ry(ry) * Rx(rx)
    """
    rx, ry, rz = euler

    cx, sx = jnp.cos(rx), jnp.sin(rx)
    cy, sy = jnp.cos(ry), jnp.sin(ry)
    cz, sz = jnp.cos(rz), jnp.sin(rz)

    # Rotation about X (roll)
    Rx = jnp.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    # Rotation about Y (pitch)
    Ry = jnp.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    # Rotation about Z (yaw)
    Rz = jnp.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


class RobotMJCFParser:
    """
    Parser for MuJoCo XML files.
    """
    _euler_seq: str = "XYZ"
    _angle_scale: float = 1.0
    _total_link_count: int = 0
    _num_actuated_joints: int = 0

    @staticmethod
    def _get_T_parent_link(element: ET.Element) -> Array:
        """
        Get the transform from the parent link to the current link.
        """
        pos = element.attrib.get("pos", "0 0 0")
        pos = [float(x) for x in pos.split()]
        pos = jnp.array(pos)

        if "quat" in element.attrib:
            quat = element.attrib["quat"]
            quat = [float(x) for x in quat.split()]
            R = mj_quat_to_rotmat(quat)
            T = jnp.eye(4)
            T = T.at[:3, :3].set(R)
            T = T.at[:3, 3].set(jnp.array(pos))
            return jaxlie.SE3.from_matrix(T)
        elif "euler" in element.attrib:
            euler = element.attrib["euler"]
            seq = RobotMJCFParser._euler_seq
            scale = RobotMJCFParser._angle_scale
            euler = {axis: float(x) for axis, x in zip(seq, euler.split())}
            euler_xyz = [euler[axis] for axis in seq]
            euler_xyz = jnp.array(euler_xyz) * scale
            R = euler_xyz_to_rotmat(euler_xyz)
            T = jnp.eye(4)
            T = T.at[:3, :3].set(R)
            T = T.at[:3, 3].set(jnp.array(pos))
            return jaxlie.SE3.from_matrix(T)
        else :
            T = jnp.eye(4)
            T = T.at[:3, 3].set(jnp.array(pos))
            return jaxlie.SE3.from_matrix(T)

    @staticmethod
    def _get_T_link_joint(element: ET.Element) -> jaxlie.SE3:
        """
        Get the transform from the link to the joint.
        """
        pos = element.attrib["pos"]
        pos = [float(x) for x in pos.split()]
        pos = jnp.array(pos)
        T = jnp.eye(4)
        T = T.at[:3, 3].set(jnp.array(pos))
        return jaxlie.SE3.from_matrix(T)

    @staticmethod
    def _get_twist_joint(element: ET.Element) -> Array:
        """
        Get the twist parameters for the joint.
        """
        joint_type = element.attrib.get("type", "hinge")
        axis = element.attrib.get("axis", "0 0 1")
        axis = [float(x) for x in axis.split()]
        axis = jnp.array(axis)
        if joint_type == "hinge":
            twist = jnp.concatenate([jnp.zeros(3), axis])
        elif joint_type == "slide":
            twist = jnp.concatenate([axis, jnp.zeros(3)])
        else:
            raise ValueError(f"Unsupported joint type {joint_type} encountered for joint '{element.attrib['name']}'.")
        assert twist.shape == (6,)
        return twist

    @staticmethod
    def _get_limits_joint(element: ET.Element) -> tuple[float, float, float]:
        """
        Get the limits for the joint.
        """
        joint_range = element.attrib.get("range", None)
        if joint_range is not None:
            joint_range = [float(x) for x in joint_range.split()]
            joint_range = jnp.array(joint_range)
            lower_limit = joint_range[0]
            upper_limit = joint_range[1]
        else:
            lower_limit = -jnp.inf
            upper_limit = jnp.inf
        return lower_limit, upper_limit, jnp.inf

    @staticmethod
    def _get_site_info(element: ET.Element) -> tuple[str, int]:
        """
        Get the site name and parent joint index.
        """
        site_name = element.attrib.get("name", None)
        site_pos = element.attrib.get("pos", "0 0 0")
        site_pos = [float(x) for x in site_pos.split()]
        site_pos = jnp.array(site_pos)
        site_euler = element.attrib.get("euler", "0 0 0")
        seq = RobotMJCFParser._euler_seq
        scale = RobotMJCFParser._angle_scale
        site_euler = {axis: float(x) for axis, x in zip(seq, site_euler.split())}
        site_euler_xyz = [site_euler[axis] for axis in seq]
        site_euler_xyz = jnp.array(site_euler_xyz) * scale
        site_R = euler_xyz_to_rotmat(site_euler_xyz)
        T = jnp.eye(4)
        T = T.at[:3, :3].set(site_R)
        T = T.at[:3, 3].set(site_pos)
        T = jaxlie.SE3.from_matrix(T)
        return site_name, T

    @staticmethod
    def _traverse(
        element: ET.Element,
        parent_link_idx: int | None,
        T_joint_link_list: list[jaxlie.SE3],
        T_parent_link_list: list[jaxlie.SE3],
        T_parent_joint_list: list[jaxlie.SE3],
        parent_indices_list: list[int],
        actuated_indices_list: list[int],
        joint_twist_list: list[Array],
        lower_limits_list: list[float],
        upper_limits_list: list[float],
        velocity_limits_list: list[float],
        lower_limits_all_list: list[float],
        upper_limits_all_list: list[float],
        velocity_limits_all_list: list[float],
        link_name_list: list[str],
        joint_name_list: list[str],
        site_name_list: list[str],
        site_parent_joint_idx_list: list[int],
        T_joint_site_list: list[jaxlie.SE3]
        ) -> None:
        """
        Traverse the XML tree and collect the information.
        """

        current_link_idx = RobotMJCFParser._total_link_count
        RobotMJCFParser._total_link_count += 1
        link_name_list.append(element.attrib.get("name", "worldbody"))
        T_parent_link_list.append(RobotMJCFParser._get_T_parent_link(element))

        has_joint = False
        joint_element: ET.Element | None = None
        for child in element:
            if child.tag == "joint" :
                assert not has_joint, "Only one joint is allowed per link"
                joint_name_list.append(child.attrib.get("name", None))
                T_joint_link_list.append(RobotMJCFParser._get_T_link_joint(child).inverse())
                has_joint = True
                joint_element = child
                limit_low, limit_high, limit_vel = RobotMJCFParser._get_limits_joint(child)
                lower_limits_list.append(limit_low)
                upper_limits_list.append(limit_high)
                velocity_limits_list.append(limit_vel)
                actuated_indices_list.append(RobotMJCFParser._num_actuated_joints)
                RobotMJCFParser._num_actuated_joints += 1

        if not has_joint :
            joint_name_list.append(None)
            T_joint_link_list.append(jaxlie.SE3.identity())
            actuated_indices_list.append(-1)
            if parent_link_idx >= 0 :
                T_parent_joint_list.append(
                    T_joint_link_list[parent_link_idx]
                    @ T_parent_link_list[current_link_idx]
                    @ T_joint_link_list[current_link_idx].inverse()
                )
            else :
                # If it is the root link, the parent joint is the world origin
                T_parent_joint_list.append(
                    T_parent_link_list[current_link_idx]
                    @ T_joint_link_list[current_link_idx].inverse()
                )
            parent_indices_list.append(parent_link_idx)
            joint_twist_list.append(jnp.zeros(6))
            lower_limits_all_list.append(0)
            upper_limits_all_list.append(0)
            velocity_limits_all_list.append(0)
        else :
            assert joint_element is not None, "Joint element expected when has_joint is True"
            if parent_link_idx >= 0:
                T_parent_joint = (
                    T_joint_link_list[parent_link_idx]
                    @ T_parent_link_list[current_link_idx]
                    @ T_joint_link_list[current_link_idx].inverse()
                )
            else:
                T_parent_joint = (
                    T_parent_link_list[current_link_idx]
                    @ T_joint_link_list[current_link_idx].inverse()
                )
            T_parent_joint_list.append(T_parent_joint)
            parent_indices_list.append(parent_link_idx)
            joint_twist_list.append(RobotMJCFParser._get_twist_joint(joint_element))
            lower_limits_all_list.append(limit_low)
            upper_limits_all_list.append(limit_high)
            velocity_limits_all_list.append(limit_vel)

        for child in element :
            if child.tag == "site" :
                site_name, T_link_site = RobotMJCFParser._get_site_info(child)
                site_name_list.append(site_name)
                site_parent_joint_idx_list.append(current_link_idx)
                T_joint_site_list.append(
                    T_joint_link_list[current_link_idx] @ T_link_site
                )

        for child in element:
            if child.tag == "body" :
                RobotMJCFParser._traverse(
                    child,
                    current_link_idx,
                    T_joint_link_list,
                    T_parent_link_list,
                    T_parent_joint_list,
                    parent_indices_list,
                    actuated_indices_list,
                    joint_twist_list,
                    lower_limits_list,
                    upper_limits_list,
                    velocity_limits_list,
                    lower_limits_all_list,
                    upper_limits_all_list,
                    velocity_limits_all_list,
                    link_name_list,
                    joint_name_list,
                    site_name_list,
                    site_parent_joint_idx_list,
                    T_joint_site_list,
                )

    @staticmethod
    def parse(xml: str) -> tuple[JointInfo, LinkInfo]:
        """
        Parses a robot kinematic tree from a XML.
        """
        if isinstance(xml, str):
            import os
            if len(xml) < 1000 and os.path.exists(xml):
                # xml is a path to a file
                tree = ET.parse(xml)
                root = tree.getroot()
            else:
                # xml is a string of XML content
                root = ET.fromstring(xml)
        else:
            # fallback: treat as file-like
            tree = ET.parse(xml)
            root = tree.getroot()

        # Lists
        T_joint_link_list = list[jaxlie.SE3]()
        T_parent_link_list = list[jaxlie.SE3]()
        T_parent_joint_list = list[jaxlie.SE3]()
        parent_indices_list = list[int]()
        actuated_indices_list = list[int]()
        joint_twist_list = list[Array]()
        lower_limits_list = list[float]()
        upper_limits_list = list[float]()
        velocity_limits_list = list[float]()
        link_name_list = list[str]()
        joint_name_list = list[str]()
        lower_limits_all_list = list[float]()
        upper_limits_all_list = list[float]()
        velocity_limits_all_list = list[float]()
        # Sites
        site_name_list = list[str]()
        site_parent_joint_idx_list = list[int]()
        T_joint_site_list = list[jaxlie.SE3]()
        # Counts
        RobotMJCFParser._total_link_count = 0
        RobotMJCFParser._num_actuated_joints = 0

        # Get compiler eulerseq
        compiler = root.find("compiler")
        if compiler is not None:
            angle_attr = compiler.attrib.get("angle", RobotMJCFParser._angle_scale)
            if angle_attr == "degree":
                RobotMJCFParser._angle_scale = jnp.pi / 180.0
            elif angle_attr == "radian":
                RobotMJCFParser._angle_scale = 1.0
            else:
                RobotMJCFParser._angle_scale = float(angle_attr)
            RobotMJCFParser._euler_seq = compiler.attrib.get(
                "eulerseq", RobotMJCFParser._euler_seq
            )

        RobotMJCFParser._traverse(
            root.find("worldbody"),
            -1,
            T_joint_link_list,
            T_parent_link_list,
            T_parent_joint_list,
            parent_indices_list,
            actuated_indices_list,
            joint_twist_list,
            lower_limits_list,
            upper_limits_list,
            velocity_limits_list,
            lower_limits_all_list,
            upper_limits_all_list,
            velocity_limits_all_list,
            link_name_list,
            joint_name_list,
            site_name_list,
            site_parent_joint_idx_list,
            T_joint_site_list,
        )

        T_parent_joint_list = [transform.wxyz_xyz for transform in T_parent_joint_list]
        actuated_name_list = [
            joint_name_list[idx] for idx, i in enumerate(actuated_indices_list) if i != -1
        ]

        joint_info = JointInfo(
            num_joints = RobotMJCFParser._total_link_count,
            num_actuated_joints = RobotMJCFParser._num_actuated_joints,
            twists = jnp.array(joint_twist_list),
            parent_transforms = jnp.array(T_parent_joint_list),
            parent_indices = jnp.array(parent_indices_list),
            actuated_indices = jnp.array(actuated_indices_list),
            lower_limits = jnp.array(lower_limits_list),
            upper_limits = jnp.array(upper_limits_list),
            velocity_limits = jnp.array(velocity_limits_list),
            names = tuple(joint_name_list),
            actuated_names = tuple(actuated_name_list),
            lower_limits_all = jnp.array(lower_limits_all_list),
            upper_limits_all = jnp.array(upper_limits_all_list),
            velocity_limits_all = jnp.array(velocity_limits_all_list),
            mimic_multiplier = jnp.ones(RobotMJCFParser._total_link_count),
            mimic_offset = jnp.zeros(RobotMJCFParser._total_link_count),
            mimic_act_indices = jnp.ones(RobotMJCFParser._total_link_count, dtype=jnp.int32) * -1,
            _topo_sort_inv = jnp.arange(RobotMJCFParser._total_link_count),
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

        # Combined links and sites
        num_links_sites = len(link_name_list) + len(site_name_list)
        names_links_sites = link_name_list + site_name_list
        parent_joint_indices = jnp.concatenate([
            jnp.arange(len(link_name_list)),
            jnp.array(site_parent_joint_idx_list, dtype=jnp.int32)
        ])
        T_parent_join_link = jnp.concatenate([
            jnp.array([transform.wxyz_xyz for transform in T_joint_link_list]),
            jnp.array([transform.wxyz_xyz for transform in T_joint_site_list]).reshape(-1, 7)
        ])
        link_info = LinkInfo(
            num_links=num_links_sites,
            names=tuple(names_links_sites),
            parent_joint_indices=parent_joint_indices,
            T_parent_joint_link=T_parent_join_link,
        )

        return joint_info, link_info
