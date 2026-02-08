"""Tests for the JAX J-PARSE implementation.

Verifies correctness of the JAX port and cross-validates against the
original numpy-based jparse_robotics package.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "examples")

from pyroki_snippets._jparse import (
    damped_least_squares,
    inverse_condition_number,
    jparse_pseudoinverse,
    jparse_step,
    manipulability_measure,
    pinv,
)


class TestJParsePseudoinverse:
    """Test the core jparse_pseudoinverse function."""

    def test_well_conditioned_2dof(self):
        """Well-conditioned 2-DOF planar Jacobian produces valid pseudo-inverse."""
        l1, l2 = 1.0, 1.0
        theta1, theta2 = np.pi / 4, np.pi / 4
        J = np.array(
            [
                [
                    -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2),
                    -l2 * np.sin(theta1 + theta2),
                ],
                [
                    l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
                    l2 * np.cos(theta1 + theta2),
                ],
            ]
        )

        J_parse, N = jparse_pseudoinverse(J, gamma=0.1)

        assert J_parse.shape == (2, 2)
        # Valid pseudo-inverse: J @ J_parse @ J ≈ J
        np.testing.assert_allclose(J @ J_parse @ J, J, atol=1e-6)
        # Well-conditioned: should be close to pinv
        np.testing.assert_allclose(J_parse, np.linalg.pinv(J), atol=0.1)

    def test_near_singular_no_nan(self):
        """Near-singular Jacobian produces finite output."""
        J = np.array([[1.0, 1.0], [1.0, 1.001]])
        J_parse, N = jparse_pseudoinverse(J, gamma=0.1)
        assert np.all(np.isfinite(J_parse))

    def test_singular_no_nan(self):
        """Exactly singular Jacobian produces finite output."""
        J = np.array([[1.0, 1.0], [1.0, 1.0]])
        J_parse, N = jparse_pseudoinverse(J, gamma=0.1)
        assert np.all(np.isfinite(J_parse))

    def test_redundant_system(self):
        """Redundant 2x3 system has correct shapes and valid pseudo-inverse."""
        J = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        J_parse, N = jparse_pseudoinverse(J, gamma=0.1)
        assert J_parse.shape == (3, 2)
        np.testing.assert_allclose(J @ J_parse @ J, J, atol=1e-6)


class TestNullspace:
    """Test nullspace projector."""

    def test_nullspace_shape(self):
        """Nullspace projector has shape (n, n)."""
        J = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        _, N = jparse_pseudoinverse(J, gamma=0.1)
        assert N.shape == (3, 3)

    def test_nullspace_projects_to_zero(self):
        """Nullspace vectors project to ~zero in task space."""
        J = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        _, N = jparse_pseudoinverse(J, gamma=0.1)
        null_vec = N @ np.array([1.0, 1.0, 1.0])
        task_effect = J @ null_vec
        np.testing.assert_allclose(task_effect, 0, atol=0.1)


class TestSingularDirectionGains:
    """Test singular direction gain parameters."""

    def test_finite_output_with_gain(self):
        """Finite output with specified singular direction gain."""
        J = np.array([[1.0, 1.0], [1.0, 1.001]])
        J_parse, _ = jparse_pseudoinverse(
            J,
            gamma=0.1,
            singular_direction_gain_position=2.0,
        )
        assert np.all(np.isfinite(J_parse))

    def test_gain_affects_output(self):
        """Different gains produce different results for near-singular case."""
        J = np.array([[1.0, 1.0], [1.0, 1.001]])
        J1, _ = jparse_pseudoinverse(
            J,
            gamma=0.1,
            singular_direction_gain_position=1.0,
        )
        J2, _ = jparse_pseudoinverse(
            J,
            gamma=0.1,
            singular_direction_gain_position=3.0,
        )
        assert not np.allclose(J1, J2)

    def test_angular_gain_affects_6dof_output(self):
        """Angular gain changes output when angular directions are singular."""
        J = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1e-4])
        J1, _ = jparse_pseudoinverse(
            J,
            gamma=0.1,
            singular_direction_gain_position=1.0,
            singular_direction_gain_angular=1.0,
            position_dimensions=3,
            angular_dimensions=3,
        )
        J2, _ = jparse_pseudoinverse(
            J,
            gamma=0.1,
            singular_direction_gain_position=1.0,
            singular_direction_gain_angular=3.0,
            position_dimensions=3,
            angular_dimensions=3,
        )
        assert not np.allclose(J1, J2)


class TestComparisonMethods:
    """Test pinv and damped_least_squares comparison methods."""

    def test_pinv_matches_numpy(self):
        """pinv matches numpy's pinv."""
        J = np.array([[1.0, 0.5], [0.5, 1.0]])
        np.testing.assert_allclose(pinv(J), np.linalg.pinv(J), atol=1e-6)

    def test_dls_formula(self):
        """DLS matches expected formula."""
        J = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = 0.1
        J_dls = damped_least_squares(J, damping=d)
        expected = np.linalg.inv(J.T @ J + d**2 * np.eye(2)) @ J.T
        np.testing.assert_allclose(J_dls, expected, atol=1e-6)


class TestMetrics:
    """Test manipulability and condition number metrics."""

    def test_manipulability_identity(self):
        """Identity Jacobian has manipulability 1."""
        J = np.eye(2)
        m = manipulability_measure(J)
        np.testing.assert_allclose(float(m), 1.0, atol=1e-6)

    def test_manipulability_near_singular_is_low(self):
        """Near-singular has lower manipulability than identity."""
        J_good = np.eye(2)
        J_bad = np.array([[1.0, 1.0], [1.0, 1.001]])
        assert float(manipulability_measure(J_bad)) < float(
            manipulability_measure(J_good)
        )

    def test_inverse_condition_identity(self):
        """Identity Jacobian has inverse condition number 1."""
        J = np.eye(2)
        np.testing.assert_allclose(float(inverse_condition_number(J)), 1.0, atol=1e-6)

    def test_inverse_condition_range(self):
        """Inverse condition number is in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            J = rng.standard_normal((3, 4))
            icn = float(inverse_condition_number(J))
            assert 0 <= icn <= 1


class TestEdgeCases:
    """Test edge cases."""

    def test_1x1_jacobian(self):
        """1x1 Jacobian produces correct pseudo-inverse."""
        J = np.array([[2.0]])
        J_parse, _ = jparse_pseudoinverse(J, gamma=0.1)
        assert J_parse.shape == (1, 1)
        np.testing.assert_allclose(float(J_parse[0, 0]), 0.5, atol=1e-6)

    def test_tall_jacobian(self):
        """Tall Jacobian (m > n) has correct shape."""
        J = np.array([[1.0], [0.5], [0.25]])
        J_parse, _ = jparse_pseudoinverse(J, gamma=0.1)
        assert J_parse.shape == (1, 3)

    def test_very_small_gamma(self):
        """Small gamma: well-conditioned case close to standard pinv."""
        J = np.array([[1.0, 0.5], [0.5, 1.0]])
        J_parse, _ = jparse_pseudoinverse(J, gamma=0.01)
        np.testing.assert_allclose(J_parse, np.linalg.pinv(J), atol=0.01)

    def test_large_gamma(self):
        """Large gamma still produces finite output."""
        J = np.array([[1.0, 0.5], [0.5, 1.0]])
        J_parse, _ = jparse_pseudoinverse(J, gamma=0.99)
        assert np.all(np.isfinite(J_parse))

    def test_missing_dimension_argument_raises(self):
        """Providing only one dimension argument raises ValueError."""
        J = np.eye(6)
        with pytest.raises(
            ValueError,
            match="Both position_dimensions and angular_dimensions must be provided.",
        ):
            jparse_pseudoinverse(J, position_dimensions=3)

    def test_dimension_sum_mismatch_raises(self):
        """Dimension sum must match Jacobian row count."""
        J = np.eye(6)
        with pytest.raises(
            ValueError,
            match="position_dimensions \\+ angular_dimensions must equal Jacobian row count.",
        ):
            jparse_pseudoinverse(J, position_dimensions=2, angular_dimensions=3)


class TestFeatureParity:
    """Cross-validate JAX implementation against original jparse_robotics."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_jparse(self):
        jparse_robotics = pytest.importorskip("jparse_robotics")
        self.JParseCore = jparse_robotics.JParseCore
        self.np_manipulability = jparse_robotics.manipulability_measure
        self.np_icn = jparse_robotics.inverse_condition_number

    @pytest.fixture()
    def test_jacobians(self):
        """Collection of test Jacobians."""
        rng = np.random.default_rng(42)
        return {
            "well_conditioned": np.array([[1.0, 0.5], [0.5, 1.0]]),
            "near_singular": np.array([[1.0, 1.0], [1.0, 1.001]]),
            "redundant": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            "random_6x7": rng.standard_normal((6, 7)),
        }

    def test_jparse_output_matches(self, test_jacobians):
        """J-PARSE output matches original for multiple test cases."""
        gamma = 0.1
        core = self.JParseCore(gamma=gamma)

        for name, J in test_jacobians.items():
            J_np = core.compute(J)
            row_count = J.shape[0]
            J_jax, _ = jparse_pseudoinverse(
                J,
                gamma=gamma,
                singular_direction_gain_position=1.0,
                singular_direction_gain_angular=1.0,
                position_dimensions=3 if row_count == 6 else row_count,
                angular_dimensions=3 if row_count == 6 else 0,
            )
            np.testing.assert_allclose(
                np.array(J_jax), np.array(J_np), atol=1e-5, err_msg=f"Failed for {name}"
            )

    def test_manipulability_matches(self, test_jacobians):
        """Manipulability measure matches original."""
        for name, J in test_jacobians.items():
            m_np = self.np_manipulability(J)
            m_jax = float(manipulability_measure(J))
            # Relaxed tolerance: JAX defaults to float32 vs numpy float64.
            np.testing.assert_allclose(
                m_jax, m_np, rtol=0.2, atol=1e-4, err_msg=f"Failed for {name}"
            )

    def test_inverse_condition_matches(self, test_jacobians):
        """Inverse condition number matches original."""
        for name, J in test_jacobians.items():
            icn_np = self.np_icn(J)
            icn_jax = float(inverse_condition_number(J))
            np.testing.assert_allclose(
                icn_jax, icn_np, atol=1e-6, err_msg=f"Failed for {name}"
            )

    def test_dls_matches(self, test_jacobians):
        """DLS matches original."""
        core = self.JParseCore(gamma=0.1)
        damping = 0.05
        for name, J in test_jacobians.items():
            J_np = core.damped_least_squares(J, damping=damping)
            J_jax = damped_least_squares(J, damping=damping)
            # Relaxed tolerance: JAX defaults to float32 vs numpy float64.
            np.testing.assert_allclose(
                np.array(J_jax), np.array(J_np), atol=1e-4, err_msg=f"Failed for {name}"
            )


class TestIntegration:
    """Integration test with pyroki robot (no Viser)."""

    @pytest.fixture()
    def panda_robot(self):
        """Load Panda robot."""
        import pyroki as pk
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        urdf = load_robot_description("panda_description")
        robot = pk.Robot.from_urdf(urdf)
        target_link_index = robot.links.names.index("panda_hand")
        return robot, target_link_index

    def test_position_error_decreases(self, panda_robot):
        """Position error decreases over several IK steps."""
        robot, target_link_index = panda_robot

        # Start from joint midpoints.
        cfg = np.array((robot.joints.lower_limits + robot.joints.upper_limits) / 2.0)

        # Target position (reachable).
        target_pos = np.array([0.5, 0.1, 0.4])

        errors = []
        for _ in range(50):
            cfg, info = jparse_step(
                robot=robot,
                cfg=cfg,
                target_link_index=target_link_index,
                target_position=target_pos,
                gamma=0.1,
                dt=0.02,
            )
            errors.append(info["position_error"])

        # Error should decrease overall.
        assert errors[-1] < errors[0]

    def test_joint_limits_respected(self, panda_robot):
        """All configs remain within joint limits."""
        robot, target_link_index = panda_robot
        cfg = np.array((robot.joints.lower_limits + robot.joints.upper_limits) / 2.0)
        target_pos = np.array([0.5, 0.1, 0.4])

        lower = np.array(robot.joints.lower_limits)
        upper = np.array(robot.joints.upper_limits)

        for _ in range(30):
            cfg, _ = jparse_step(
                robot=robot,
                cfg=cfg,
                target_link_index=target_link_index,
                target_position=target_pos,
                dt=0.02,
            )
            assert np.all(cfg >= lower - 1e-6)
            assert np.all(cfg <= upper + 1e-6)

    def test_info_dict_keys(self, panda_robot):
        """Info dict contains expected keys."""
        robot, target_link_index = panda_robot
        cfg = np.array((robot.joints.lower_limits + robot.joints.upper_limits) / 2.0)
        target_pos = np.array([0.5, 0.1, 0.4])

        _, info = jparse_step(
            robot=robot,
            cfg=cfg,
            target_link_index=target_link_index,
            target_position=target_pos,
        )
        expected_keys = {
            "position_error",
            "orientation_error",
            "max_joint_vel",
            "jacobian",
            "manipulability",
            "inverse_condition_number",
        }
        assert set(info.keys()) == expected_keys

    def test_orientation_tracking(self, panda_robot):
        """Orientation tracking uses 6-DOF Jacobian and reduces error."""
        import jaxlie

        robot, target_link_index = panda_robot
        cfg = np.array((robot.joints.lower_limits + robot.joints.upper_limits) / 2.0)

        # Use a small orientation offset from the initial EE pose so
        # the controller can converge in limited steps.
        import jax.numpy as jnp

        poses = robot.forward_kinematics(jnp.asarray(cfg))
        init_pose = jaxlie.SE3(poses[target_link_index])
        target_pos = np.array(init_pose.translation()) + np.array([0.05, 0.0, 0.0])
        target_wxyz = np.array(init_pose.rotation().wxyz)

        errors = []
        for _ in range(100):
            cfg, info = jparse_step(
                robot=robot,
                cfg=cfg,
                target_link_index=target_link_index,
                target_position=target_pos,
                target_wxyz=target_wxyz,
                gamma=0.1,
                dt=0.02,
            )
            errors.append(info["position_error"])

        # Position error should decrease.
        assert errors[-1] < errors[0]
        # Jacobian should be 6xn when tracking orientation.
        assert info["jacobian"].shape[0] == 6
        # Orientation error should be present and finite.
        assert np.isfinite(info["orientation_error"])

    def test_orientation_convergence_large_rotation(self, panda_robot):
        """Orientation converges with a non-trivial (~45 deg) rotation offset.

        This exercises the geometric Jacobian path: the analytical Jacobian
        introduces artificial singularities at large rotations that prevent
        convergence, while the geometric Jacobian handles them correctly.
        """
        import jax.numpy as jnp
        import jaxlie

        robot, target_link_index = panda_robot
        cfg = np.array((robot.joints.lower_limits + robot.joints.upper_limits) / 2.0)

        poses = robot.forward_kinematics(jnp.asarray(cfg))
        init_pose = jaxlie.SE3(poses[target_link_index])

        # Keep position the same, apply a ~45-degree rotation about the z-axis.
        target_pos = np.array(init_pose.translation())
        angle = np.pi / 4  # 45 degrees
        offset_rot = jaxlie.SO3.from_z_radians(angle)
        target_rot = offset_rot @ init_pose.rotation()
        target_wxyz = np.array(target_rot.wxyz)

        ori_errors = []
        for _ in range(200):
            cfg, info = jparse_step(
                robot=robot,
                cfg=cfg,
                target_link_index=target_link_index,
                target_position=target_pos,
                target_wxyz=target_wxyz,
                gamma=0.1,
                dt=0.02,
            )
            ori_errors.append(info["orientation_error"])

        # Orientation error should converge to a small value.
        assert ori_errors[-1] < 0.15, (
            f"Orientation error did not converge: {ori_errors[-1]:.4f}"
        )
