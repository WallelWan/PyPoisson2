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

## 安装
```bash
pip install -e .
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
测试基于 `unittest`，以脚本方式运行（不是 pytest）。

```bash
python poisson/tests/test_poisson.py
PYTHONPATH=. python poisson/tests/test_phase2_boundary.py
PYTHONPATH=. python poisson/tests/test_phase3_advanced_outputs.py
```

注意：
- `poisson/tests/test_phase3_advanced_outputs.py` 运行时间较长。
- 如无法导入 `poisson`，请确保在仓库根目录执行并设置 `PYTHONPATH=.`。
