"""Residual functions for pyroki costs and constraints.

These functions have the same signature as cost/constraint functions
(taking `vals: VarValues` as first arg), so they can be directly wrapped:

    limit_cost = Cost.create_factory(limit_residual)
    limit_constraint = Constraint.create_factory(limit_residual, constraint_type="leq_zero")
"""

from ._pose_residual_analytic_jac import (
    pose_cost_analytic_jac as pose_cost_analytic_jac,
)
from ._pose_residual_numerical_jac import (
    pose_cost_numerical_jac as pose_cost_numerical_jac,
)
from ._residuals import (
    five_point_acceleration_residual as five_point_acceleration_residual,
)
from ._residuals import five_point_jerk_residual as five_point_jerk_residual
from ._residuals import five_point_velocity_residual as five_point_velocity_residual
from ._residuals import limit_acceleration_residual as limit_acceleration_residual
from ._residuals import limit_jerk_residual as limit_jerk_residual
from ._residuals import limit_residual as limit_residual
from ._residuals import limit_velocity_residual as limit_velocity_residual
from ._residuals import manipulability_residual as manipulability_residual
from ._residuals import pose_residual as pose_residual
from ._residuals import pose_with_base_residual as pose_with_base_residual
from ._residuals import rest_residual as rest_residual
from ._residuals import rest_with_base_residual as rest_with_base_residual
from ._residuals import self_collision_residual as self_collision_residual
from ._residuals import smoothness_residual as smoothness_residual
from ._residuals import world_collision_residual as world_collision_residual
