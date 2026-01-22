# pypoisson2

基于 Screened Poisson Surface Reconstruction 的 Python 绑定，支持网格场、密度与梯度输出。

## 功能
- 通过 C++ 库进行泊松重建（自动选择 int32/int64）
- 可选隐式场网格输出
- 支持边界条件（neumann/dirichlet/free）
- 可选密度与梯度输出

## 环境要求
- Python 3.9+
- NumPy
- 已编译的 PoissonRecon 共享库，位于 `poisson/lib/`、`lib/` 或 `build/`
- C++ 构建依赖：`cmake`, `g++`, `make`, `zlib`, `libpng`, `libjpeg`, OpenMP

## 安装
```bash
pip install -e .
```

## 快速开始
```bash
# 构建共享库
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
make -C build -j$(nproc)

# 安装 Python 包
pip install -e .

# 运行基础重建示例
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

## 构建（PoissonRecon + Dump 工具）
本仓库构建两个可执行程序：
- `PoissonRecon`（输出目录：`build/`）
- `PoissonReconDump`（输出目录：`build_dump/`，用于导出 density/gradients）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
make -C build -j$(nproc)

cmake -S poisson_dump -B build_dump -DCMAKE_BUILD_TYPE=Release
make -C build_dump -j$(nproc)
```

注意：如需完整边界/degree 支持，请不要定义 `FAST_COMPILE`。

## PyPI 发布（维护者）
```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

## 基础用法
```python
import numpy as np
from poisson import poisson_reconstruction

points = np.random.randn(1000, 3).astype(np.float64)
points /= np.linalg.norm(points, axis=1, keepdims=True)

normals = points.copy()

vertices, faces = poisson_reconstruction(points, normals, depth=6)
```

## 高级输出
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

## 核心参数
`poisson_reconstruction` 暴露核心求解参数：
- `depth`, `full_depth`, `cg_depth`, `iters`, `degree`
- `scale`, `samples_per_node`, `point_weight`, `confidence`
- `exact_interpolation`, `show_residual`, `low_depth_cutoff`
- `width`, `cg_solver_accuracy`, `base_depth`, `solve_depth`, `kernel_depth`, `base_v_cycles`
- `parallel_type`, `grid_depth`
- `validate_finite`（默认 True）, `force_big`（强制 int64 库）

## 测试
测试基于 `unittest`，以模块方式运行（不是 pytest）。

```bash
python -m unittest poisson.tests.exe.test_exe_consistency_core
python -m unittest poisson.tests.exe.test_exe_consistency_params
python -m unittest poisson.tests.exe.test_exe_boundary
python -m unittest poisson.tests.exe.test_dump_outputs
python -m unittest poisson.tests.python.test_python_validation
```

可执行文件路径（可选覆盖）：
- `POISSON_EXE_PATH` 默认 `build/PoissonRecon`
- `POISSON_DUMP_EXE_PATH` 默认 `build_dump/PoissonReconDump`

注意：
- 边界测试默认使用 horse 数据：`poisson/examples/horse_with_normals.xyz`。
- `PoissonReconDump` 只支持 ASCII PLY 输入（x y z nx ny nz）。
