"""Cost factories for pyroki optimization.

These are created by wrapping residual functions with Cost.create_factory.
"""

from jaxls import Cost

from ._residuals import (
    five_point_acceleration_residual,
    five_point_jerk_residual,
    five_point_velocity_residual,
    limit_residual,
    limit_velocity_residual,
    manipulability_residual,
    pose_cost_analytic_jac,
    pose_cost_numerical_jac,
    pose_residual,
    pose_with_base_residual,
    rest_residual,
    rest_with_base_residual,
    self_collision_residual,
    smoothness_residual,
    world_collision_residual,
)

# Pose costs
pose_cost = Cost.create_factory(pose_residual)
pose_cost_with_base = Cost.create_factory(pose_with_base_residual)

# Specialized pose costs with custom Jacobians (already wrapped)
pose_cost_analytic_jac = pose_cost_analytic_jac
pose_cost_numerical_jac = pose_cost_numerical_jac

# Limit costs
limit_cost = Cost.create_factory(limit_residual)
limit_velocity_cost = Cost.create_factory(limit_velocity_residual)

# Regularization costs
rest_cost = Cost.create_factory(rest_residual)
rest_with_base_cost = Cost.create_factory(rest_with_base_residual)
smoothness_cost = Cost.create_factory(smoothness_residual)

# Manipulability cost
manipulability_cost = Cost.create_factory(manipulability_residual)

# Collision costs
self_collision_cost = Cost.create_factory(self_collision_residual)
world_collision_cost = Cost.create_factory(world_collision_residual)

# Finite difference costs
five_point_velocity_cost = Cost.create_factory(five_point_velocity_residual)
five_point_acceleration_cost = Cost.create_factory(five_point_acceleration_residual)
five_point_jerk_cost = Cost.create_factory(five_point_jerk_residual)
