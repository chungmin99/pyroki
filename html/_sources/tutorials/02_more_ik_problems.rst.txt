Diving Deeper into IK
==========================================

This section expands on the :doc:`basic IK example <01_quickstart>` by extending it to:

1. Multiple joint targets (e.g., for bimanual robots).
2. Simultaneously optimizing the robot base pose with the joint configuration (e.g., for mobile manipulators).
3. Incorporating collision avoidance costs.
4. Optimizing for robot end-effector manipulability.

All the examples below are available as snippits in :mod:`pyroki_snippets`, and interactive demos are located in the ``examples/`` directory.


==========================================
IK with multiple joint targets
==========================================

- *The demo is available in `02_bimanual_ik.py`.*
- *The implementation is available as* :func:`solve_ik_with_multiple_targets` *in* :mod:`pyroki_snippets`.

We show how ``pyroki`` can be used for IK problems with multiple pose targets, through a bimanual IK with the ABB YuMI robot.
The setup mirrors the basic IK example, but leverages the fact that :func:`pk.costs.pose_cost` can handle multiple targets.

.. code-block:: python

    # Load the robot.
    urdf = load_robot_description("yumi_description")
    robot = pk.Robot.from_urdf(urdf)

    # Define the EE link names and the target poses.
    ee_link_names = ["yumi_link_7_r", "yumi_link_7_l"]
    target_positions = np.array(...)  # shape: (2, 3)
    target_orientations = np.array(...)  # shape: (2, 4)

    # Define the variable -- a single robot joint configuration.
    joint_var = robot.joint_var_cls(0)

    # Define the costs.
    link_indices = [robot.links.names.index(name) for name in ee_link_names]
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_ori_wxyz),
                target_position
            ), # from SE3 with batch dimension (1,) -> (2,).
            link_indices,  # from link_index (1,) -> (2,).
            pos_weight=1.0,
            ori_weight=1.0,
        ),
        pk.costs.joint_limits(robot, joint_var, weight=100.0),
    ]

    ...  # Continue as before.

==========================================
IK with Mobile Robot Base
==========================================

- *The demo is available in `03_mobile_ik.py`.*
- *The implementation is available as* :func:`solve_ik_with_base` *in* :mod:`pyroki_snippets`.

``pyroki`` can also handle IK problems with mobile robot bases; the setup is similar to the original single-arm IK example, but with an additional variable for the base pose.

.. code-block:: python

    # Load the robot.
    urdf = load_robot_description("fetch_description")
    robot = pk.Robot.from_urdf(urdf)

    # Define the EE link names and the target poses.
    ee_link_name = "gripper_link"
    target_position = np.array(...)  # shape: (3,)
    target_orientation_wxyz = np.array(...)  # shape: (4,)

    # Define the variables (plural this time)!
    joint_var = robot.joint_var_cls(0)
    base_pos_var = jaxls.SE3Var(0)

    # Define the costs.
    link_index = robot.links.names.index(ee_link_name)
    costs = [
        pk.costs.pose_cost_with_base(
            robot,
            joint_var,
            base_var,  # <- this cost takes in two variables.
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_orientation_wxyz),
                target_position,
            ),
            link_index,
            pos_weight=1.0,
            ori_weight=1.0,
        ),
        pk.costs.limit_cost(robot, joint_var, weight=100.0),
    ]

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var, base_var])  # Both variables!
        .analyze()
        .solve()
    )
    joint_cfg = sol[joint_var]


==========================================
IK with Collision Avoidance
==========================================

- *The demo is available in `04_collision_ik.py`.*
- *The implementation is available as* :func:`solve_ik_with_collision` *in* :mod:`pyroki_snippets`.

Collision avoidance is a common requirement for robots, so ``pyroki`` provides helpers for this purpose.

We can define a :class:`pk.collision.RobotCollision` object from the URDF for all things robot-collision related, including self-collisions and world-collisions.

.. code-block:: python

    # Load the robot.
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    world_coll: pk.collision.CollGeom = ...

    joint_var = robot.joint_var_cls(0)
    target_link_index = robot.links.names.index(end_effector_link)

    costs = [
        ... # Pose cost, limit cost from the basic IK example.
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            joint_var,
            margin=0.02,  # this tunes the collision distance margin.
            weight=5.0,
        ),
        pk.costs.world_collision_cost(
            robot,
            robot_coll,
            joint_var,
            world_coll,
            margin=0.05,
            weight=10.0,
        ),
    ]

    ...  # Continue as before.


We can also extend the collision logic for sweeping volumes -- e.g., to check robot collisions *between* timesteps, which we will go into depth in :doc:`later <03_trajopt>`.

==========================================
IK with Manipulability
==========================================

- *The demo is available in `05_manipulability_ik.py`.*
- *The implementation is available as* :func:`solve_ik_with_manipulability` *in* :mod:`pyroki_snippets`.

We can also incorporate task-space manipulability into the IK objective,
by maximizing the Yoshikawa manipulability index for the specified end-effector!

.. code-block:: python

    # Load the robot.
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Define the EE link name and get its index.
    target_link_name = "panda_hand"  # As used in the demo
    target_link_index = robot.links.names.index(target_link_name)

    # Define the joint variable.
    joint_var = robot.joint_var_cls(0)

    # Define the costs.
    costs = [
        ... # Pose cost, limit cost, rest cost.
        pk.costs.manipulability_cost(
            robot,
            joint_var,
            target_link_index,
            weight=1.0,
        ),
    ]

    ...  # Continue as before with problem setup and solving.

The manipulability cost here :func:`pk.costs.manipulability_cost` is equivalent to:

.. code-block:: python

    @pk.Cost.create_factory
    def manipulability_cost(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        joint_var: jaxls.Var[jnp.ndarray],
        target_link_index: jnp.ndarray,
        manipulability_weight: jnp.ndarray | float,
    ):
        cfg = vals[joint_var]
        
        # Calculate the Yoshikawa manipulability index.
        jacobian = jax.jacfwd(
            lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
        )(cfg)[target_link_index]
        JJT = jacobian @ jacobian.T
        assert JJT.shape == (3, 3)
        
        # Cost is inverse of the index --> we maximize it.
        return 1.0 / (jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(JJT))) + 1e-6)



Now, we've covered how we can take an existing objective, and add new costs to it.

*Splendid!*