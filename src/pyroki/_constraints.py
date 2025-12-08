"""Constraint factories for pyroki optimization.

These are created by wrapping residual functions with Constraint.create_factory.

Constraint semantics:
- Inequality constraints (leq_zero): g(x) <= 0 means satisfied
- Residuals are positive when violated, negative/zero when satisfied
"""

from jaxls import Constraint

from ._residuals import (
    limit_residual,
    limit_velocity_residual,
    self_collision_residual,
    world_collision_residual,
)

# Limit constraints
limit_constraint = Constraint.create_factory(
    limit_residual, constraint_type="leq_zero"
)
limit_velocity_constraint = Constraint.create_factory(
    limit_velocity_residual, constraint_type="leq_zero"
)

# Collision constraints
world_collision_constraint = Constraint.create_factory(
    world_collision_residual, constraint_type="leq_zero"
)
