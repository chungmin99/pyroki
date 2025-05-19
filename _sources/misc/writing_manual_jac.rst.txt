:orphan:

Defining Jacobians Manually
=====================================

``pyroki`` supports both autodiff and manually defined Jacobians for computing cost gradients. 

For reference, this is the robot pose matching cost :math:`C_\text{pose}`:

.. math::

    \sum_{i} \left( w_{p,i} \left\| \mathbf{p}_{i}(q) - \mathbf{p}_{i}^* \right\|^2 + w_{R,i} \left\| \text{log}(\mathbf{R}_{i}(q)^{-1} \mathbf{R}_{i}^*) \right\|^2 \right)


where :math:`q` is the robot joint configuration, :math:`\mathbf{p}_{i}(q)` is the position of the :math:`i`-th link, :math:`\mathbf{R}_{i}(q)` is the rotation matrix of the :math:`i`-th link, and :math:`w_{p,i}` and :math:`w_{R,i}` are the position and orientation weights, respectively.

The following is the most common way to define costs in ``pyroki`` -- with autodiff:

.. code-block:: python

    @Cost.create_factory
    def pose_cost(
        vals: VarValues,
        robot: Robot,
        joint_var: Var[Array],
        target_pose: jaxlie.SE3,
        target_link_index: Array,
        pos_weight: Array | float,
        ori_weight: Array | float,
    ) -> Array:
        """Computes the residual for matching link poses to target poses."""
        assert target_link_index.dtype == jnp.int32
        joint_cfg = vals[joint_var]
        Ts_link_world = robot.forward_kinematics(joint_cfg)
        pose_actual = jaxlie.SE3(Ts_link_world[..., target_link_index, :])

        # Position residual = position error * weight
        pos_residual = (pose_actual.translation() - target_pose.translation()) * pos_weight
        # Orientation residual = log(actual_inv * target) * weight
        ori_residual = (pose_actual.rotation().inverse() @ target_pose.rotation()).log() * ori_weight

        return jnp.concatenate([pos_residual, ori_residual]).flatten()

The alternative is to manually write out the Jacobian -- while automatic differentiation is convenient and works well for most use cases, analytical Jacobians can provide better performance, which we show in the `paper <https://arxiv.org/abs/2505.03728>`_.

We provide two implementations of pose matching cost with custom Jacobians:

- an `analytically derived Jacobian <https://github.com/chungmin99/pyroki/blob/main/src/pyroki/costs/_pose_cost_analytic_jac.py>`_ (~200 lines), or
- a `numerically computed Jacobian <https://github.com/chungmin99/pyroki/blob/main/src/pyroki/costs/_pose_cost_numerical_jac.py>`_ through finite differences (~50 lines).