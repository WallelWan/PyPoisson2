# pypoisson2

Python bindings for Screened Poisson Surface Reconstruction with optional grid output, density values, and gradient vectors.

## Features
- Poisson surface reconstruction via C++ libraries (int32/int64 auto-selection)
- Optional implicit grid field output
- Boundary condition support (neumann/dirichlet/free)
- Optional density and gradient outputs

## Requirements
- Python 3.9+
- NumPy
- Compiled PoissonRecon shared libraries in `poisson/lib/`, `lib/`, or `build/`

## Installation
```bash
pip install -e .
```

## Basic Usage
```python
import numpy as np
from poisson import poisson_reconstruction

points = np.random.randn(1000, 3).astype(np.float64)
points /= np.linalg.norm(points, axis=1, keepdims=True)

normals = points.copy()

vertices, faces = poisson_reconstruction(points, normals, depth=6)
```

## Advanced Outputs
```python
vertices, faces, grid, iso_value, densities, gradients = poisson_reconstruction(
    points,
    normals,
    depth=6,
    grid_depth=5,
    output_density=True,
    output_gradients=True,
)
```

## Core Parameters
`poisson_reconstruction` exposes core solver parameters:
- `depth`, `full_depth`, `cg_depth`, `iters`, `degree`
- `scale`, `samples_per_node`, `point_weight`, `confidence`
- `exact_interpolation`, `show_residual`, `low_depth_cutoff`
- `width`, `cg_solver_accuracy`, `base_depth`, `solve_depth`, `kernel_depth`, `base_v_cycles`
- `parallel_type`, `grid_depth`
- `validate_finite` (default True), `force_big` (force int64 library)

## Tests
Tests use `unittest` and are run as scripts (not pytest).

```bash
python poisson/tests/test_poisson.py
PYTHONPATH=. python poisson/tests/test_phase2_boundary.py
PYTHONPATH=. python poisson/tests/test_phase3_advanced_outputs.py
```

Notes:
- `poisson/tests/test_phase3_advanced_outputs.py` is long-running and may take several minutes.
- If `poisson` cannot be imported, ensure `PYTHONPATH=.` (or run from repo root).
