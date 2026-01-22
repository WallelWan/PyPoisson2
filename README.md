# pypoisson2

A new Python bindings for Poisson Reconstruction with optional grid output, density values, and gradient vectors.

The corresponding C++ version: v18.74 of [PoissonRecon](https://github.com/mkazhdan/PoissonRecon) by [mkazhdan](https://github.com/mkazhdan).

## Features
- Poisson surface reconstruction via C++ libraries (int32/int64 auto-selection)
- Optional implicit grid field output
- Boundary condition support (neumann/dirichlet/free)
- Optional density and gradient outputs

## Requirements
- Python 3.9+
- NumPy
- Compiled PoissonRecon shared libraries in `poisson/lib/`, `lib/`, or `build/`
- C++ build deps: `cmake`, `g++`, `make`, `zlib`, `libpng`, `libjpeg`, OpenMP

## Installation
```bash
pip install -e .
```

## Quick Start
```bash
# Build shared libs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
make -C build -j$(nproc)

# Install Python package
pip install -e .

# Run a basic reconstruction
python - <<'PY'
import numpy as np
from poisson import poisson_reconstruction

points = np.random.randn(1000, 3).astype(np.float64)
points /= np.linalg.norm(points, axis=1, keepdims=True)
normals = points.copy()

vertices, faces = poisson_reconstruction(points, normals, depth=6)
print(vertices.shape, faces.shape)
PY
```

## Build (PoissonRecon + Dump Tool)
This repo builds two executables:
- `PoissonRecon` in `build/`
- `PoissonReconDump` in `build_dump/` (exports density/gradients for test alignment)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
make -C build -j$(nproc)

cmake -S poisson_dump -B build_dump -DCMAKE_BUILD_TYPE=Release
make -C build_dump -j$(nproc)
```

Note: Do not define `FAST_COMPILE` if you need full boundary/degree support.

## PyPI Release (Maintainers)
```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
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
Tests use `unittest` and are run as modules (not pytest).

```bash
python -m unittest poisson.tests.exe.test_exe_consistency_core
python -m unittest poisson.tests.exe.test_exe_consistency_params
python -m unittest poisson.tests.exe.test_exe_boundary
python -m unittest poisson.tests.exe.test_dump_outputs
python -m unittest poisson.tests.python.test_python_validation
```

Executable paths (optional overrides):
- `POISSON_EXE_PATH` defaults to `build/PoissonRecon`
- `POISSON_DUMP_EXE_PATH` defaults to `build_dump/PoissonReconDump`

Notes:
- Boundary tests default to the horse dataset: `poisson/examples/horse_with_normals.xyz`.
- `PoissonReconDump` supports ASCII PLY inputs (x y z nx ny nz).
