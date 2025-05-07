Hello, Inverse Kinematics
==========================================

Let's start with a "Hello World" example for PyRoki: solving the inverse kinematics (IK) problem for the Franka Panda robot arm (7 DoF).

First, let's load the robot.

.. code-block:: python
    
    # Load the robot description.
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    urdf = load_robot_description("panda_description")

    # Create the robot object.
    import pyroki as pk
    robot = pk.Robot.from_urdf(urdf)


Now, let's solve the IK problem using the :func:`solve_ik` snippit available in :mod:`pyroki_snippets`.

.. code-block:: python

    # Define the end-effector link name and its target pose.
    end_effector_link = "panda_hand"
    assert end_effector_link in robot.links.names

    import numpy as np
    target_position = np.array([0.7, 0.0, 0.5])
    target_orientation_wxyz = np.array([0.0, 0.0, 1.0, 0.0])  # pointing downwards

    # Solve!
    import pyroki_snippets as pks
    solution_cfg = pks.solve_ik(
        robot=robot,
        target_link_name=end_effector_link,
        target_position=target_position,
        target_orientation_wxyz=target_orientation_wxyz,
    )

And that's it!
    
The file `01_basic_ik.py` contains an interactive demo using the same :func:`solve_ik` function, where you can experiment with different target poses and visualize the results in real-time.

.. video:: /_static/basic_ik.mov
    :width: 100%
    :nocontrols:
    :autoplay:
    :playsinline:
    :muted:
    :loop:


==========================================
Defining the IK problem from scratch
==========================================

Alternatively, we can define the IK problem from scratch as a nonlinear least-squares optimization problem, specifying the variables and costs.

Our only variable here is the robot joint configuration (we will go more into the variable's syntax :doc:`later <03_trajopt>`).

.. code-block:: python

    joint_var = robot.joint_var_cls(0)

The two costs to optimize are: pose matching and joint limits.

.. code-block:: python

    import jaxlie  # Library for Lie algebra operations.

    target_link_index = robot.links.names.index(end_effector_link)

    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_ori_wxyz), target_position
            ),
            target_link_index,
            pos_weight=1.0,
            ori_weight=1.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
    ]

Then, we solve the IK problem defined by these variables and costs.

.. code-block:: python

    import jaxls  # Library for nonlinear least-squares optimization.

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve()
    )
    solution_cfg = sol[joint_var]


Now we've implemented the inverse kinematics problem... *great*... and now onwards!