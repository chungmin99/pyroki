# `PyRoki`: Python Robot Kinematics Library

**[Project page](https://pyroki-toolkit.github.io/) &bull;
[arXiv](https://arxiv.org/abs/2505.03728)**

`PyRoki` is a modular, extensible, and cross-platform toolkit for kinematic optimization, all in Python.

Core features include:

- Differentiable robot forward kinematics model from a URDF.
- Automatic generation of robot collision primitives (e.g., capsules).
- Differentiable collision bodies with numpy broadcasting logic.
- Common cost implementations (e.g., end effector pose, self/world-collision, manipulability).
- Arbitrary costs, autodiff or analytical Jacobians.
- Integration with a [Levenberg-Marquardt Solver](https://github.com/brentyi/jaxls) that supports optimization on manifolds (e.g., [lie groups](https://github.com/brentyi/jaxlie))
- Cross-platform support (CPU, GPU, TPU) via JAX.

Please refer to the [documentation](https://chungmin99.github.io/pyroki/) for more details, features, and usage examples.

---

## Installation

You can install `pyroki` with `pip`, on Python 3.10+:

```
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```

## Status

_May 6, 2025_: Initial release

We are preparing and will release by _May 16, 2025_:

- [x] Examples + documentation for hand / humanoid motion retargeting
- [x] Documentation for using manually defined Jacobians
- [x] Support with Python 3.10+

## Limitations

- **Soft constraints only**: We use a nonlinear least-squares formulation and model joint limits, collision avoidance, etc. as soft penalties with high weights rather than hard constraints.
- **Static shapes & JIT overhead**: JAX JIT compilation is triggered on first run and when input shapes change (e.g., number of targets, obstacles). Arrays can be pre-padded to vectorize over inputs with different shapes.
- **No sampling-based planners**: We don't include sampling-based planners (e.g., graphs, trees).
- **Collision performance**: Speed and accuracy comparisons against other robot toolkits such as CuRobo have not been extensively performed, and is likely slower than other toolkits for collision-heavy scenarios.

The following are current implementation limitations that could potentially be addressed in future versions:

- **Joint types**: We only support revolute, continuous, prismatic, and fixed joints. Other URDF joint types are treated as fixed joints.
- **Collision geometry**: We are limited to sphere, capsule, halfspace, and heightmap geometries. Mesh collision is approximated as capsules.
- **Kinematic structures**: We only support kinematic trees; no closed-loop mechanisms or parallel manipulators.

## Citation

This codebase is released with the following preprint.

<table><tr><td>
    Chung Min Kim*, Brent Yi*, Hongsuk Choi, Yi Ma, Ken Goldberg, Angjoo Kanazawa.
    <strong>PyRoki: A Modular Toolkit for Robot Kinematic Optimization</strong>
    arXiV, 2025.
</td></tr>
</table>

<sup>\*</sup><em>Equal Contribution</em>, <em>UC Berkeley</em>.

Please cite PyRoki if you find this work useful for your research:

```
@inproceedings{kim2025pyroki,
  title={PyRoki: A Modular Toolkit for Robot Kinematic Optimization},
  author={Kim*, Chung Min and Yi*, Brent and Choi, Hongsuk and Ma, Yi and Goldberg, Ken and Kanazawa, Angjoo},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025},
  url={https://arxiv.org/abs/2505.03728},
}
```

Thanks!
