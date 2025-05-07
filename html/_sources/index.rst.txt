PyRoki
==========

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
- Arbitrary costs, getting Jacobians either calculated through autodiff or defined manually.
- Integration with a `Levenberg-Marquardt Solver <https://github.com/brentyi/jaxls>`_.
- Cross-platform support (CPU, GPU, TPU).



Installation
------------

You can install ``pyroki`` with ``pip``, with Python 3.12+:

.. code-block:: bash

   git clone https://github.com/chungmin99/pyroki.git
   cd pyroki
   pip install -e .


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

   @article{pyroki2025,
   author = {Kim, Chung Min* and Yi, Brent* and Choi, Hongsuk and Ma, Yi and Goldberg, Ken and Kanazawa, Angjoo},
   title = {PyRoki: A Modular Toolkit for Robot Kinematic Optimization},
   journal = {arXiv},
   year = {2025},
   } 

Thanks for using ``PyRoki``!