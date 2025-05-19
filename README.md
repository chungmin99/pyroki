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
- Integration with a [Levenberg-Marquardt Solver](https://github.com/brentyi/jaxls).
- Cross-platform support (CPU, GPU, TPU).

Please refer to the [documentation](https://chungmin99.github.io/pyroki/) for more details, features, and usage examples.

---

## Installation

You can install `pyroki` with `pip`, on Python 3.12+:

```
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```

Python 3.10-3.11 should also work, but support may be dropped in the future.

## Status

_May 6, 2025_: Initial release

We are preparing and will release by _May 16, 2025_:

- [x] Examples + documentation for hand / humanoid motion retargeting
- [ ] Documentation for using manually defined Jacobians.
- [x] Support with Python 3.10+

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
@article{pyroki2025,
 author = {Kim, Chung Min* and Yi, Brent* and Choi, Hongsuk and Ma, Yi and Goldberg, Ken and Kanazawa, Angjoo},
 title = {PyRoki: A Modular Toolkit for Robot Kinematic Optimization},
 journal = {arXiv},
 year = {2025},
}
```

Thanks!
