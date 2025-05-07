Trajectory Optimization
==========================================

``pyroki`` supports trajectory optimization -- a natural extension of the inverse kinematics problem.


From IK to trajectory optimization
------------------------------------------

First, let's extend the :doc:`basic IK <01_quickstart>` example to multiple timesteps.

Let's replace the single joint variable from the IK example with a *sequence* of joint variables, representing the configuration at each of the ``num_timesteps``:

.. code-block:: python

    # before: joint_var = robot.joint_var_cls(0)
    joint_vars = robot.joint_var_cls(np.arange(num_timesteps))

.. note::
   Variables are defined by the variable type (``SE3Var``) and the variable index (``0``) -- a pointer-like syntax.
   
   The indexing notation supports both integers (``Var(0)``), or a sequence of integers (``Var(jnp.arange(N))``).


Then, the costs can take in this sequence of variables:

.. code-block:: python

    import jaxlie  # Library for Lie algebra operations.

    target_link_index = robot.links.names.index(end_effector_link)
    target_poses = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_ori_wxyz), target_position
    )  # SE3 poses, of batch size (num_timesteps,)

    # Create batched costs.
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_vars,
            target_poses,
            target_link_index,
            pos_weight=1.0,
            ori_weight=1.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_vars,
            weight=100.0,
        ),
    ]

    # Solve the multi-timestep IK problem.
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_vars])
        .analyze()
        .solve()
    )
    solution_cfg = sol[joint_vars]  # (num_timesteps, num_actuated_joints)



A-to-B trajectory planning
------------------------------------------

- *The demo is available in `05_trajopt.py`.*
- *The implementation is available as* :func:`solve_trajopt` *in* :mod:`pyroki_snippets`.

In the previous section we assumed there exists a target pose for *each* timestep, but in practice we often want to plan a trajectory from an initial configuration to a final configuration.
Here, we will solve for a collision-free trajectory that penalizes large joint acceleration and jerk.

This is an abridged version of the implementation:

.. code-block:: python

    # Load the robot, and create its collision model.
    urdf = load_robot_description("ur5_description")
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    
    # Define the world collision geometry.
    world_coll: pk.collision.CollGeom = ...

    # Solve for collision-free configurations for the start and end EE poses.
    start_cfg, end_cfg = solve_iks_with_collision(...)

    # Create variables.
    traj_vars = robot.joint_var_cls(jnp.arange(num_timesteps))

    start_var = robot.joint_var_cls(0)
    end_var = robot.joint_var_cls(num_timesteps - 1)

    traj_vars_prev = robot.joint_var_cls(jnp.arange(num_timesteps - 1))
    traj_vars_next = robot.joint_var_cls(jnp.arange(1, num_timesteps))

    # Create batched costs.
    costs = [
        # Point-level costs (e.g., start / end pose).
        pk.costs.joint_similarity_cost(robot, start_var, start_cfg),
        pk.costs.joint_similarity_cost(robot, end_var, end_cfg),
        ...,

        # Trajectory-level costs (e.g., smoothness, limits, jerk).
        pk.costs.smoothness_cost(robot, traj_vars_prev, traj_vars_next),
        pk.costs.limit_cost(robot, traj_vars),
        pk.costs.five_point_acceleration_cost(...),
        
        # World collision costs, with continuous collision checking.
        world_collision_cost_with_volume_sweep(traj_vars_prev, traj_vars_next, ...),
        ...
    ]

    # Solve.
    ...

Some notes regarding batch notations:

#. The first joint configuration is supervised by *both* the start pose cost and the trajectory-level costs (similar to pointer notation).
#. The smoothness cost is applied pairwise between consecutive timesteps (similar to array programming). One way to think about it is that we feed in ``fun(Var([0, 1]), Var([1, 2]))``, and we evaluate ``fun(Var([0]), Var([1]))`` and ``fun(Var([1]), Var([2]))``.

Also, note that:

#. Only the first and last joint variables are supervised explicitly with a configuration matching cost.
#. The ``world_collision_cost_with_volume_sweep`` function is a custom cost that computes the collision distance between the robot and the world geometry, using a volume-swept approach between consecutive timesteps. 
