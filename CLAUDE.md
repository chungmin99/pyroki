# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyRoki is a JAX-based Python library for differentiable robot kinematics and optimization. It provides forward kinematics, collision detection, and cost/constraint factories that integrate with the `jaxls` nonlinear least-squares solver. Python 3.10+.

## Common Commands

### Install
```bash
sudo apt install -y libsuitesparse-dev  # system dependency for scikit-sparse
pip install -e ".[dev]"
```

### Test
```bash
pytest                    # run all tests
pytest tests/test_ik_vmap.py  # run single test file
pytest -k "test_name"     # run specific test by name
```

### Lint & Format
```bash
ruff check docs/ src/ examples/        # lint
ruff format docs/ src/ examples/       # format
ruff format docs/ src/ examples/ --check  # check formatting without modifying
```

### Type Check
```bash
pyright .
```

### Build Docs
```bash
cd docs && sphinx-build source build -b dirhtml
```

## Architecture

### Core Pipeline

The typical workflow is: **URDF → Robot → forward kinematics → residuals → costs → jaxls solver**.

1. **`Robot`** (`_robot.py`): `jax_dataclasses.pytree_dataclass` holding joint/link info. Created via `Robot.from_urdf()`. The `forward_kinematics(cfg)` method returns SE(3) transforms (as wxyz-xyz arrays) for all links. Supports `jax.vmap` for batched configs.

2. **Residuals** (`_residuals/`): Functions with signature `(vals: VarValues, robot, joint_var, ...) -> Array`. These compute optimization residuals (pose error, joint limits, collision distances, etc.).

3. **Costs** (`costs.py`): Thin wrappers created via `Cost.factory(residual_fn)`. Costs are soft penalties (least squares); constraints use `kind="constraint_leq_zero"` for augmented Lagrangian enforcement.

4. **Collision** (`collision/`): `RobotCollision.from_urdf()` builds collision geometry (spheres, capsules, boxes). `CollGeom` is the base class for geometry primitives. `collide()` computes signed distances between geometry pairs.

5. **Solver**: Uses `jaxls` (external) for Levenberg-Marquardt optimization with optional augmented Lagrangian constraints. Problems are built with `jaxls.LeastSquaresProblem` and solved with `jaxls.solve()`.

### Key Patterns

- All core data structures are `@jdc.pytree_dataclass` (JAX-compatible pytrees via `jax_dataclasses`)
- SE(3) transforms use `jaxlie.SE3` (Lie group library)
- Typed arrays use `jaxtyping.Float[Array, "shape"]` annotations
- Variables for optimization use `jaxls.Var` subclasses; `Robot.joint_var_cls` is auto-created from URDF
- `@jdc.jit` is used instead of `@jax.jit` for methods on pytree dataclasses

### Examples Structure

Examples live in `examples/` and use helper modules from `examples/pyroki_snippets/` for reusable solve patterns (e.g., `solve_ik`, `solve_ik_with_collision`). Visualization uses `viser`.

## CI

Three workflows run on PRs to `main`:
- **pytest**: Tests across Python 3.10–3.13
- **pyright**: Type checking across Python 3.10–3.13
- **formatting**: `ruff check` + `ruff format --check` on `docs/`, `src/`, `examples/`
