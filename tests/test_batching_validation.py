"""
Test suite for batching validation system in PyRoki.

This test suite validates that the robot batching detection and validation
system works correctly across all entry points and provides helpful error
messages when users incorrectly batch robot structures.
"""

import jax
import jax.numpy as jnp
import pytest
from pyroki import Robot
from pyroki.collision import RobotCollision
from robot_descriptions.loaders.yourdfpy import load_robot_description


class TestBatchingValidation:
    """Test suite for the batching validation system."""

    @pytest.fixture
    def robot(self):
        """Create a test robot from URDF."""
        urdf = load_robot_description("panda_description")
        return Robot.from_urdf(urdf)

    @pytest.fixture
    def robot_batched(self, robot):
        """Create an incorrectly batched robot for testing validation."""
        return jax.tree.map(lambda x: x[None], robot)

    @pytest.fixture
    def valid_config(self, robot):
        """Create a valid configuration for the robot."""
        return jnp.zeros(robot.joints.num_actuated_joints)

    @pytest.fixture
    def valid_configs_batch(self, robot):
        """Create a batch of valid configurations."""
        return jnp.zeros((5, robot.joints.num_actuated_joints))

    def test_normal_robot_passes_validation(self, robot):
        """Test that a normal robot passes all validation checks."""
        # Should not raise any exceptions
        robot.assert_no_batched()
        robot.joints.assert_no_batched() 
        robot.links.assert_no_batched()

    def test_normal_robot_forward_kinematics(self, robot, valid_config, valid_configs_batch):
        """Test that normal robot works with forward kinematics."""
        # Single configuration
        result_single = robot.forward_kinematics(valid_config)
        assert result_single.shape == (robot.links.num_links, 7)
        
        # Batched configurations  
        result_batch = robot.forward_kinematics(valid_configs_batch)
        assert result_batch.shape == (5, robot.links.num_links, 7)

    def test_normal_robot_get_full_config(self, robot, valid_config, valid_configs_batch):
        """Test that normal robot works with get_full_config."""
        # Single configuration
        result_single = robot.joints.get_full_config(valid_config)
        assert result_single.shape == (robot.joints.num_joints,)
        
        # Batched configurations
        result_batch = robot.joints.get_full_config(valid_configs_batch)
        assert result_batch.shape == (5, robot.joints.num_joints)

    def test_vmap_operations_work(self, robot, valid_configs_batch):
        """Test that vmap operations work correctly with normal robot."""
        result = jax.vmap(robot.forward_kinematics)(valid_configs_batch)
        assert result.shape == (5, robot.links.num_links, 7)

    def test_batched_robot_fails_validation(self, robot_batched):
        """Test that batched robot fails validation with helpful error."""
        with pytest.raises(AssertionError) as exc_info:
            robot_batched.assert_no_batched()
        
        error_msg = str(exc_info.value)
        assert "Joint twists has unexpected shape" in error_msg
        assert "This suggests the robot structure has been batched" in error_msg
        assert "Use batched configurations instead of batching the robot" in error_msg

    def test_batched_robot_joints_validation(self, robot_batched):
        """Test that batched robot joints fail validation."""
        with pytest.raises(AssertionError) as exc_info:
            robot_batched.joints.assert_no_batched()
        
        error_msg = str(exc_info.value)
        assert "Joint twists has unexpected shape" in error_msg
        assert f"expected ({robot_batched.joints.num_joints}, 6)" in error_msg

    def test_batched_robot_links_validation(self, robot_batched):
        """Test that batched robot links fail validation."""
        with pytest.raises(AssertionError) as exc_info:
            robot_batched.links.assert_no_batched()
        
        error_msg = str(exc_info.value)
        assert "Link parent_joint_indices has unexpected shape" in error_msg
        assert f"expected ({robot_batched.links.num_links},)" in error_msg

    def test_batched_robot_forward_kinematics_fails(self, robot_batched, valid_config):
        """Test that forward kinematics fails on batched robot."""
        with pytest.raises(AssertionError) as exc_info:
            robot_batched.forward_kinematics(valid_config)
        
        error_msg = str(exc_info.value)
        assert "Joint twists has unexpected shape" in error_msg

    def test_batched_robot_get_full_config_fails(self, robot_batched, valid_config):
        """Test that get_full_config fails on batched robot."""
        with pytest.raises(AssertionError) as exc_info:
            robot_batched.joints.get_full_config(valid_config)
        
        error_msg = str(exc_info.value)
        assert "Joint twists has unexpected shape" in error_msg

    def test_input_shape_validation(self, robot):
        """Test that input shape validation works correctly."""
        # Wrong input shape should fail with helpful message
        wrong_config = jnp.zeros(robot.joints.num_actuated_joints + 1)
        
        with pytest.raises(AssertionError) as exc_info:
            robot.forward_kinematics(wrong_config)
        
        error_msg = str(exc_info.value)
        assert "Configuration shape mismatch" in error_msg
        assert f"Expected (*batch, {robot.joints.num_actuated_joints})" in error_msg

    def test_actuated_config_shape_validation(self, robot):
        """Test that actuated config shape validation works in get_full_config."""
        wrong_config = jnp.zeros(robot.joints.num_actuated_joints + 1)
        
        with pytest.raises(AssertionError) as exc_info:
            robot.joints.get_full_config(wrong_config)
        
        error_msg = str(exc_info.value)
        assert "Actuated configuration shape mismatch" in error_msg
        assert f"Expected (*batch, {robot.joints.num_actuated_joints})" in error_msg

    def test_all_joint_arrays_validated(self, robot):
        """Test that all joint structural arrays are validated."""
        robot_batched = jax.tree.map(lambda x: x[None], robot)
        
        # Check that the validation covers all the expected arrays
        expected_errors = [
            "twists", "parent_transforms", "parent_indices", "actuated_indices",
            "mimic_act_indices", "mimic_multiplier", "mimic_offset", "_topo_sort_inv"
        ]
        
        # At least one of these should be caught (they all get batched together)
        with pytest.raises(AssertionError) as exc_info:
            robot_batched.joints.assert_no_batched()
        
        error_msg = str(exc_info.value)
        # Should mention one of the joint arrays
        assert any(array_name in error_msg for array_name in expected_errors)

    def test_error_message_quality(self, robot):
        """Test that error messages are helpful and actionable."""
        robot_batched = jax.tree.map(lambda x: x[None], robot)
        
        with pytest.raises(AssertionError) as exc_info:
            robot_batched.assert_no_batched()
        
        error_msg = str(exc_info.value)
        
        # Should explain the problem
        assert "unexpected shape" in error_msg
        assert "expected" in error_msg
        
        # Should explain the likely cause
        assert "robot structure has been batched" in error_msg
        
        # Should provide solution
        assert "Use batched configurations instead of batching the robot" in error_msg

    def test_validation_at_multiple_entry_points(self, robot, valid_config):
        """Test that validation works at multiple API entry points."""
        robot_batched = jax.tree.map(lambda x: x[None], robot)
        
        # All of these should fail with validation errors
        entry_points = [
            lambda: robot_batched.forward_kinematics(valid_config),
            lambda: robot_batched.joints.get_full_config(valid_config),
            lambda: robot_batched.assert_no_batched(),
            lambda: robot_batched.joints.assert_no_batched(),
            lambda: robot_batched.links.assert_no_batched(),
        ]
        
        for entry_point in entry_points:
            with pytest.raises(AssertionError):
                entry_point()

    def test_batch_dimensions_detected_correctly(self, robot):
        """Test that different types of batching are detected correctly."""
        # Single batch dimension
        robot_1d_batch = jax.tree.map(lambda x: x[None], robot)
        with pytest.raises(AssertionError):
            robot_1d_batch.assert_no_batched()
        
        # Multiple batch dimensions
        robot_2d_batch = jax.tree.map(lambda x: x[None, None], robot)
        with pytest.raises(AssertionError):
            robot_2d_batch.assert_no_batched()

    def test_normal_operations_unaffected(self, robot, valid_config, valid_configs_batch):
        """Test that normal operations are unaffected by validation."""
        # These should all work without any performance impact
        
        # Single config operations
        fk_single = robot.forward_kinematics(valid_config)
        full_config_single = robot.joints.get_full_config(valid_config)
        
        # Batch config operations  
        fk_batch = robot.forward_kinematics(valid_configs_batch)
        full_config_batch = robot.joints.get_full_config(valid_configs_batch)
        
        # VMapped operations
        fk_vmap = jax.vmap(robot.forward_kinematics)(valid_configs_batch)
        
        # Verify results have expected shapes
        assert fk_single.shape == (robot.links.num_links, 7)
        assert fk_batch.shape == (5, robot.links.num_links, 7)
        assert fk_vmap.shape == (5, robot.links.num_links, 7)
        assert full_config_single.shape == (robot.joints.num_joints,)
        assert full_config_batch.shape == (5, robot.joints.num_joints)


class TestCollisionSystemIntegration:
    """Test that collision system works correctly with validation."""

    @pytest.fixture
    def robot(self):
        """Create a test robot from URDF."""
        urdf = load_robot_description("panda_description")
        return Robot.from_urdf(urdf)

    @pytest.fixture 
    def robot_coll(self):
        """Create a collision model for the robot."""
        urdf = load_robot_description("panda_description")
        return RobotCollision.from_urdf(urdf)

    def test_collision_vmap_works(self, robot, robot_coll):
        """Test that collision system works with vmap (correct approach)."""
        traj = jnp.zeros((5, robot.joints.num_actuated_joints))
        
        # This should work fine
        swept_capsules = jax.vmap(
            lambda x, y: robot_coll.get_swept_capsules(robot, x, y),
            in_axes=(0, 0),
        )(traj[:-1, :], traj[1:, :])
        
        # Should be able to convert to trimesh
        trimesh_result = swept_capsules.to_trimesh()
        assert trimesh_result is not None


if __name__ == "__main__":
    # Allow running the test file directly for quick validation
    pytest.main([__file__, "-v"])