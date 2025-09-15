PyRoki
==========

`Project page <https://pyroki-toolkit.github.io/>`_ `•` `arXiv <https://arxiv.org/abs/2505.03728>`_ `•` `Code <https://github.com/chungmin99/pyroki>`_

**PyRoki** is a library for robot kinematic optimization (Python Robot Kinematics).

1. **Modular**: Optimization variables and cost functions are decoupled, enabling reusable components across tasks. Objectives like collision avoidance and pose matching can be applied to both IK and trajectory optimization without reimplementation.

2. **Extensible**: ``PyRoki`` supports automatic differentiation for user-defined costs with Jacobian computation, a real-time cost-weight tuning interface, and optional analytical Jacobians for performance-critical use cases.

3. **Cross-Platform**: ``PyRoki`` runs on CPU, GPU, and TPU, allowing efficient scaling from single-robot use cases to large-scale parallel processing for motion datasets or planning.

We demonstrate how ``PyRoki`` solves IK, trajectory optimization, and motion retargeting for robot hands and humanoids in a unified framework. It uses a Levenberg-Marquardt optimizer to efficiently solve these tasks, and we evaluate its performance on batched IK.

Features include:

- Differentiable robot forward kinematics model from a URDF.
- Automatic generation of robot collision primitives (e.g., capsules).
- Differentiable collision bodies with numpy broadcasting logic. 
- Common cost factors (e.g., end effector pose, self/world-collision, manipulability).
- Arbitrary costs, getting Jacobians either calculated :doc:`through autodiff or defined manually<misc/writing_manual_jac>`.
- Integration with a `Levenberg-Marquardt Solver <https://github.com/brentyi/jaxls>`_ that supports optimization on manifolds (e.g., `lie groups <https://github.com/brentyi/jaxlie>`_).
- Cross-platform support (CPU, GPU, TPU) via JAX.



Installation
------------

You can install ``pyroki`` with ``pip``, on Python 3.12+:

.. code-block:: bash

   git clone https://github.com/chungmin99/pyroki.git
   cd pyroki
   pip install -e .


Python 3.10-3.11 should also work, but support may be dropped in the future.

Limitations
-----------

- **Soft constraints only**: We use a nonlinear least-squares formulation and model joint limits, collision avoidance, etc. as soft penalties with high weights rather than hard constraints.
- **Static shapes & JIT overhead**: JAX JIT compilation is triggered on first run and when input shapes change (e.g., number of targets, obstacles). Arrays can be pre-padded to vectorize over inputs with different shapes.
- **No sampling-based planners**: We don't include sampling-based planners (e.g., graphs, trees).
- **Collision performance**: Speed and accuracy comparisons against other robot toolkits such as CuRobo have not been extensively performed, and is likely slower than other toolkits for collision-heavy scenarios.

The following are current implementation limitations that could potentially be addressed in future versions:

- **Joint types**: We only support revolute, continuous, prismatic, and fixed joints. Other URDF joint types are treated as fixed joints.
- **Collision geometry**: We are limited to sphere, capsule, halfspace, and heightmap geometries. Mesh collision is approximated as capsules.
- **Kinematic structures**: We only support kinematic chains; no closed-loop mechanisms or parallel manipulators.

Examples
--------

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/01_basic_ik
   examples/02_bimanual_ik
   examples/03_mobile_ik
   examples/04_ik_with_coll
   examples/05_ik_with_manipulability
   examples/06_online_planning
   examples/07_trajopt
   examples/08_ik_with_mimic_joints
   examples/09_hand_retargeting
   examples/10_humanoid_retargeting
   examples/11_hand_retargeting_fancy
   examples/12_humanoid_retargeting_fancy


Acknowledgements
----------------
``PyRoki`` is heavily inspired by the prior work, including but not limited to 
`Trac-IK <https://traclabs.com/projects/trac-ik/>`_,
`cuRobo <https://curobo.org>`_,
`pink <https://github.com/stephane-caron/pink>`_,
`mink <https://github.com/kevinzakka/mink>`_,
`Drake <https://drake.mit.edu/>`_, and 
`Dex-Retargeting <https://github.com/dexsuite/dex-retargeting>`_.
Thank you so much for your great work!


Citation
--------

If you find this work useful, please cite it as follows:

.. code-block:: bibtex

   @inproceedings{kim2025pyroki,
      title={PyRoki: A Modular Toolkit for Robot Kinematic Optimization},
      author={Kim*, Chung Min and Yi*, Brent and Choi, Hongsuk and Ma, Yi and Goldberg, Ken and Kanazawa, Angjoo},
      booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2025},
      url={https://arxiv.org/abs/2505.03728},
   }

Thanks for using ``PyRoki``!
