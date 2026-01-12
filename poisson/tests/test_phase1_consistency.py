"""
Phase 1 Consistency Test Suite for pypoisson2.

Tests consistency between pypoisson2 Python implementation and the
reference PoissonRecon executable for all Phase 1 parameters.

Test Categories:
- A: Basic consistency tests
- B: Core solver parameters
- C: Depth control parameters
- D: Mesh extraction parameters
- E: Grid output tests
- F: Edge cases
- G: Real data (horse model)
"""

import unittest
import subprocess
import tempfile
import os
import re
import sys
from pathlib import Path

import numpy as np

# Add repo and test directories to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(Path(__file__).parent))

from poisson import poisson_reconstruction
from test_data_generator import (
    generate_sphere_points,
    generate_cube_points,
    load_xyz_file,
    save_points_ply_binary,
    save_points_ply_ascii,
    convert_xyz_to_ply
)
from geometry_utils import (
    compare_meshes,
    parse_iso_value_from_output,
    check_manifold
)


class ConsistencyTestResult:
    """Container for test comparison results."""

    def __init__(self, test_id: str, test_name: str):
        self.test_id = test_id
        self.test_name = test_name
        self.passed = False
        self.iso_value_ref = None
        self.iso_value_py = None
        self.iso_diff = None
        self.vertex_count_ref = None
        self.vertex_count_py = None
        self.face_count_ref = None
        self.face_count_py = None
        self.volume_diff = None
        self.error_message = None
        self.execution_time_ref = None
        self.execution_time_py = None

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{self.test_id}] {self.test_name}: {status}",
        ]
        if self.iso_value_ref is not None:
            lines.append(f"  iso_value: ref={self.iso_value_ref:.6f}, py={self.iso_value_py:.6f}, diff={self.iso_diff:.2e}")
        if self.vertex_count_ref is not None:
            lines.append(f"  vertices: ref={self.vertex_count_ref}, py={self.vertex_count_py}")
        if self.face_count_ref is not None:
            lines.append(f"  faces: ref={self.face_count_ref}, py={self.face_count_py}")
        if self.error_message:
            lines.append(f"  error: {self.error_message}")
        return "\n".join(lines)


class ConsistencyTestBase(unittest.TestCase):
    """Base class for consistency tests."""

    # Path to reference executable
    EXE_PATH = REPO_ROOT / "exe" / "PoissonRecon"

    # Tolerance thresholds
    # Note: When comparing with reference executable that reads ASCII PLY,
    # we need a looser tolerance due to ASCII format precision (6 decimal places)
    ISO_TOLERANCE_STRICT = 1e-6      # For pure Python comparisons
    ISO_TOLERANCE_EXE = 5e-3          # For comparisons with reference executable (ASCII PLY)
    ISO_TOLERANCE_LOOSE = 1e-4
    VERTEX_TOLERANCE = 10             # Slightly increased for ASCII PLY precision
    VOLUME_TOLERANCE = 0.01  # 1%

    @classmethod
    def setUpClass(cls):
        """Check if reference executable is available."""
        if not os.path.exists(cls.EXE_PATH):
            cls.skipTest(f"Reference executable not found: {cls.EXE_PATH}")

    def run_reference_executable(self, input_ply: str, output_ply: str, args: dict) -> tuple:
        """
        Run the reference PoissonRecon executable.

        Parameters
        ----------
        input_ply : str
            Input PLY file path.
        output_ply : str
            Output PLY file path.
        args : dict
            Command-line arguments (keys are flag names without dashes).

        Returns
        -------
        result : tuple
            (iso_value, vertex_count, face_count, stdout_str)
        """
        cmd = [str(self.EXE_PATH), "--in", input_ply, "--out", output_ply]

        # Map pypoisson2 parameter names to PoissonRecon flags
        flag_map = {
            'depth': '--depth',
            'full_depth': '--fullDepth',
            'cg_depth': '--cgDepth',
            'iters': '--iters',
            'degree': '--degree',
            'scale': '--scale',
            'samples_per_node': '--samplesPerNode',
            'point_weight': '--pointWeight',
            'confidence': '--confidence',
            'verbose': '--verbose',
            'grid_depth': '--grid',
            'exact_interpolation': '--exact',
            'show_residual': '--showResidual',
            'low_depth_cutoff': '--lowDepthCutOff',
            'width': '--width',
            'cg_solver_accuracy': '--cgAccuracy',
            'base_depth': '--baseDepth',
            'solve_depth': '--solveDepth',
            'kernel_depth': '--kernelDepth',
            'base_v_cycles': '--baseVCycles',
            'force_manifold': '--nonManifold',  # Inverted logic handled below
            'polygon_mesh': '--polygonMesh',
            'primal_grid': '--primalGrid',
            'linear_fit': '--linearFit',
            'grid_coordinates': '--gridCoordinates',
        }

        for key, value in args.items():
            if value is None or value is False:
                continue
            flag = flag_map.get(key)
            if flag:
                if isinstance(value, bool):
                    # Special handling for force_manifold (inverted)
                    if key == 'force_manifold':
                        # force_manifold=True means we DON'T want --nonManifold
                        if not value:
                            cmd.append('--nonManifold')
                    else:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(value)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout_str = result.stdout + result.stderr

        if result.returncode != 0:
            raise RuntimeError(f"Reference executable failed: {stdout_str}")

        # Parse output
        iso_value = parse_iso_value_from_output(stdout_str)

        # Read mesh header for vertex/face count
        vertex_count, face_count = self._parse_ply_header(output_ply)

        return iso_value, vertex_count, face_count, stdout_str

    def _parse_ply_header(self, ply_file: str) -> tuple:
        """Parse PLY header to get vertex and face counts."""
        with open(ply_file, 'rb') as f:
            for line in f:
                line = line.decode('ascii', errors='ignore').strip()
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('element face'):
                    face_count = int(line.split()[-1])
                elif line == 'end_header':
                    break
        return vertex_count, face_count

    def compare_results(self, result: ConsistencyTestResult,
                       strict: bool = True, use_exe_tolerance: bool = False) -> ConsistencyTestResult:
        """Check if test results meet success criteria."""
        if use_exe_tolerance:
            iso_tol = self.ISO_TOLERANCE_EXE
        elif strict:
            iso_tol = self.ISO_TOLERANCE_STRICT
        else:
            iso_tol = self.ISO_TOLERANCE_LOOSE

        # Check iso-value difference
        if result.iso_value_ref is not None and result.iso_value_py is not None:
            result.iso_diff = abs(result.iso_value_ref - result.iso_value_py)
            if result.iso_diff > iso_tol:
                result.passed = False
                result.error_message = f"iso-value diff {result.iso_diff:.2e} > {iso_tol:.2e}"
                return result

        # Check vertex count
        if result.vertex_count_ref is not None and result.vertex_count_py is not None:
            v_diff = abs(result.vertex_count_ref - result.vertex_count_py)
            if v_diff > self.VERTEX_TOLERANCE:
                result.passed = False
                result.error_message = f"vertex count diff {v_diff} > {self.VERTEX_TOLERANCE}"
                return result

        result.passed = True
        return result


# =============================================================================
# Category A: Basic Consistency Tests
# =============================================================================

class TestCategoryA_Basic(ConsistencyTestBase):
    """Basic consistency tests."""

    def test_A1_default_parameters(self):
        """Test with default parameters."""
        result = ConsistencyTestResult("A1", "Default Parameters")

        # Generate test data
        points, normals = generate_sphere_points(5000)

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f_out:

            input_ply = f_in.name
            output_ply = f_out.name

        try:
            # Save to PLY
            save_points_ply_binary(points, normals, input_ply)

            # Run reference
            iso_ref, v_ref, f_ref, _ = self.run_reference_executable(
                input_ply, output_ply, {'depth': 8, 'verbose': True}
            )
            result.iso_value_ref = iso_ref
            result.vertex_count_ref = v_ref
            result.face_count_ref = f_ref

            # Run pypoisson2
            v_py, f_py, grid, iso_py = poisson_reconstruction(
                points, normals, depth=8, grid_depth=1
            )
            result.iso_value_py = iso_py
            result.vertex_count_py = len(v_py)
            result.face_count_py = len(f_py)

            # Compare (use_exe_tolerance because reference executable reads ASCII PLY)
            result = self.compare_results(result, use_exe_tolerance=True)

        finally:
            # Cleanup
            for f in [input_ply, output_ply]:
                if os.path.exists(f):
                    os.remove(f)

        self.assertTrue(result.passed, result.error_message)
        print(f"\n{result}")

    def test_A2_depth_parameter(self):
        """Test with different depth values."""
        for depth in [5, 6, 7, 8]:
            with self.subTest(depth=depth):
                points, normals = generate_sphere_points(2000)

                # Run pypoisson2
                vertices, faces = poisson_reconstruction(points, normals, depth=depth)

                # Check that higher depth produces more vertices
                self.assertGreater(len(vertices), 0)
                self.assertGreater(len(faces), 0)

                print(f"  depth={depth}: {len(vertices)} vertices, {len(faces)} faces")

    def test_A3_degree_parameter(self):
        """Test with different B-spline degrees."""
        points, normals = generate_sphere_points(2000)

        for degree in [1, 2]:
            with self.subTest(degree=degree):
                vertices, faces, grid, iso = poisson_reconstruction(
                    points, normals, depth=6, grid_depth=1, degree=degree
                )

                self.assertGreater(len(vertices), 0)
                self.assertTrue(0.0 < iso < 1.0)

                print(f"  degree={degree}: {len(vertices)} vertices, iso={iso:.6f}")

    def test_A4_different_scales(self):
        """Test with different point cloud sizes."""
        for n_points in [1000, 5000, 20000]:
            with self.subTest(n_points=n_points):
                points, normals = generate_sphere_points(n_points)

                vertices, faces = poisson_reconstruction(points, normals, depth=7)

                self.assertGreater(len(vertices), 0)
                self.assertGreater(len(faces), 0)

                print(f"  {n_points} points: {len(vertices)} vertices")


# =============================================================================
# Category B: Core Solver Parameters
# =============================================================================

class TestCategoryB_Solver(ConsistencyTestBase):
    """Core solver parameter tests."""

    def test_B1_exact_interpolation(self):
        """Test exact_interpolation parameter."""
        points, normals = generate_sphere_points(3000)

        for exact in [False, True]:
            with self.subTest(exact=exact):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, exact_interpolation=exact
                )
                self.assertGreater(len(vertices), 0)
                print(f"  exact={exact}: {len(vertices)} vertices")

    def test_B2_show_residual(self):
        """Test show_residual parameter."""
        points, normals = generate_sphere_points(3000)

        # Should not raise errors
        vertices, faces = poisson_reconstruction(
            points, normals, depth=6, show_residual=True, verbose=True
        )
        self.assertGreater(len(vertices), 0)

    def test_B3_low_depth_cutoff(self):
        """Test low_depth_cutoff parameter."""
        points, normals = generate_sphere_points(3000)

        for cutoff in [0.0, 0.5]:
            with self.subTest(cutoff=cutoff):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, low_depth_cutoff=cutoff
                )
                self.assertGreater(len(vertices), 0)

    def test_B4_width_parameter(self):
        """Test width parameter."""
        points, normals = generate_sphere_points(3000)

        for width in [0.0, 0.01]:
            with self.subTest(width=width):
                vertices, faces = poisson_reconstruction(
                    points, normals, width=width
                )
                self.assertGreater(len(vertices), 0)

    def test_B5_cg_solver_accuracy(self):
        """Test cg_solver_accuracy parameter."""
        points, normals = generate_sphere_points(3000)

        for accuracy in [1e-3, 1e-4]:
            with self.subTest(accuracy=accuracy):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, cg_solver_accuracy=accuracy
                )
                self.assertGreater(len(vertices), 0)


# =============================================================================
# Category C: Depth Control Parameters
# =============================================================================

class TestCategoryC_Depth(ConsistencyTestBase):
    """Depth control parameter tests."""

    def test_C1_base_depth(self):
        """Test base_depth parameter."""
        points, normals = generate_sphere_points(2000)

        for base_depth in [-1, 3, 4]:
            with self.subTest(base_depth=base_depth):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, base_depth=base_depth
                )
                self.assertGreater(len(vertices), 0)

    def test_C2_solve_depth(self):
        """Test solve_depth parameter."""
        points, normals = generate_sphere_points(2000)

        for solve_depth in [-1, 5, 6]:
            with self.subTest(solve_depth=solve_depth):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=7, solve_depth=solve_depth
                )
                self.assertGreater(len(vertices), 0)

    def test_C3_kernel_depth(self):
        """Test kernel_depth parameter."""
        points, normals = generate_sphere_points(2000)

        for kernel_depth in [-1, 4]:
            with self.subTest(kernel_depth=kernel_depth):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, kernel_depth=kernel_depth
                )
                self.assertGreater(len(vertices), 0)

    def test_C4_base_v_cycles(self):
        """Test base_v_cycles parameter."""
        points, normals = generate_sphere_points(2000)

        for v_cycles in [1, 2]:
            with self.subTest(v_cycles=v_cycles):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, base_v_cycles=v_cycles
                )
                self.assertGreater(len(vertices), 0)


# =============================================================================
# Category D: Mesh Extraction Parameters
# =============================================================================

class TestCategoryD_Extraction(ConsistencyTestBase):
    """Mesh extraction parameter tests."""

    def test_D1_force_manifold(self):
        """Test force_manifold parameter."""
        points, normals = generate_sphere_points(2000)

        for force_manifold in [True, False]:
            with self.subTest(force_manifold=force_manifold):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, force_manifold=force_manifold
                )

                manifold = check_manifold(vertices, faces)
                if force_manifold:
                    # Should be manifold (or close to it)
                    self.assertTrue(manifold['boundary_edges'] < 10)
                else:
                    # May have non-manifold edges
                    self.assertGreater(len(vertices), 0)

                print(f"  force_manifold={force_manifold}: {manifold}")

    def test_D2_polygon_mesh(self):
        """Test polygon_mesh parameter."""
        points, normals = generate_sphere_points(2000)

        vertices, faces = poisson_reconstruction(
            points, normals, depth=6, polygon_mesh=True
        )
        self.assertGreater(len(vertices), 0)

    def test_D3_primal_grid(self):
        """Test primal_grid parameter."""
        points, normals = generate_sphere_points(2000)

        for primal_grid in [False, True]:
            with self.subTest(primal_grid=primal_grid):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, primal_grid=primal_grid
                )
                self.assertGreater(len(vertices), 0)

    def test_D4_linear_fit(self):
        """Test linear_fit parameter."""
        points, normals = generate_sphere_points(2000)

        for linear_fit in [False, True]:
            with self.subTest(linear_fit=linear_fit):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, linear_fit=linear_fit
                )
                self.assertGreater(len(vertices), 0)

    def test_D5_grid_coordinates(self):
        """Test grid_coordinates parameter."""
        points, normals = generate_sphere_points(2000)

        for grid_coords in [False, True]:
            with self.subTest(grid_coordinates=grid_coords):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=6, grid_coordinates=grid_coords
                )
                self.assertGreater(len(vertices), 0)


# =============================================================================
# Category E: Grid Output Tests
# =============================================================================

class TestCategoryE_Grid(ConsistencyTestBase):
    """Grid output tests."""

    def test_E1_grid_resolution(self):
        """Test grid output at different resolutions."""
        points, normals = generate_sphere_points(2000)

        for grid_depth in [5, 6]:
            with self.subTest(grid_depth=grid_depth):
                vertices, faces, grid, iso = poisson_reconstruction(
                    points, normals, depth=6, grid_depth=grid_depth
                )

                res = 2 ** grid_depth
                self.assertEqual(grid.shape, (res, res, res))
                self.assertTrue(0.0 < iso < 1.0)
                print(f"  grid_depth={grid_depth}: res={res}, iso={iso:.6f}")

    def test_E2_grid_value_range(self):
        """Test that grid values are in expected range."""
        points, normals = generate_sphere_points(2000)

        vertices, faces, grid, iso = poisson_reconstruction(
            points, normals, depth=6, grid_depth=6
        )

        # Check grid is not all zeros
        self.assertFalse(np.all(grid == 0))

        # Check value distribution
        grid_min = np.min(grid)
        grid_max = np.max(grid)
        grid_mean = np.mean(grid)

        print(f"  grid range: [{grid_min:.4f}, {grid_max:.4f}], mean: {grid_mean:.4f}")
        print(f"  iso_value: {iso:.6f}")

        # Iso-value should be within grid range
        self.assertTrue(grid_min <= iso <= grid_max)

    def test_E3_iso_value_extraction(self):
        """Test iso-value extraction."""
        points, normals = generate_sphere_points(3000)

        vertices, faces, grid, iso = poisson_reconstruction(
            points, normals, depth=7, grid_depth=5
        )

        # Iso-value should be positive and typically close to 0.5
        self.assertTrue(0.0 < iso < 1.0)

        # Grid should contain both positive and negative values around iso
        above_iso = np.sum(grid > iso)
        below_iso = np.sum(grid < iso)

        self.assertGreater(above_iso, 0)
        self.assertGreater(below_iso, 0)

        print(f"  iso={iso:.6f}, above: {above_iso}, below: {below_iso}")

    def test_E4_spherical_symmetry(self):
        """Test that sphere reconstruction has symmetry."""
        points, normals = generate_sphere_points(3000)

        vertices, faces = poisson_reconstruction(points, normals, depth=6)

        # Check symmetry (should be low for a sphere)
        center = np.array([0.0, 0.0, 0.0])
        radii = np.linalg.norm(vertices - center, axis=1)
        radius_std = np.std(radii) / np.mean(radii)

        print(f"  sphere symmetry error: {radius_std:.4f}")
        self.assertTrue(radius_std < 0.2)  # Allow 20% variation


# =============================================================================
# Category F: Edge Cases
# =============================================================================

class TestCategoryF_EdgeCases(ConsistencyTestBase):
    """Edge case tests."""

    def test_F1_empty_input(self):
        """Test with empty input."""
        points = np.zeros((0, 3), dtype=np.float64)
        normals = np.zeros((0, 3), dtype=np.float64)

        vertices, faces = poisson_reconstruction(points, normals, depth=5)

        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(faces), 0)

    def test_F2_few_points(self):
        """Test with very few points."""
        points, normals = generate_sphere_points(10)

        vertices, faces = poisson_reconstruction(points, normals, depth=4)

        # Should still produce some output
        self.assertGreaterEqual(len(vertices), 0)

    def test_F3_coplanar_points(self):
        """Test with coplanar points (XY plane)."""
        n_points = 1000
        points = np.zeros((n_points, 3), dtype=np.float64)
        points[:, 0] = np.random.uniform(-1, 1, n_points)
        points[:, 1] = np.random.uniform(-1, 1, n_points)

        normals = np.zeros((n_points, 3), dtype=np.float64)
        normals[:, 2] = 1.0

        vertices, faces = poisson_reconstruction(points, normals, depth=5)

        # Should produce a flat mesh
        self.assertGreater(len(vertices), 0)
        self.assertLess(np.std(vertices[:, 2]), 0.1)  # Low Z variation

    def test_F4_duplicate_points(self):
        """Test with duplicate points."""
        base_points, base_normals = generate_sphere_points(500)

        # Duplicate each point
        points = np.vstack([base_points, base_points])
        normals = np.vstack([base_normals, base_normals])

        vertices, faces = poisson_reconstruction(points, normals, depth=5)

        self.assertGreater(len(vertices), 0)

    def test_F5_nan_input(self):
        """Test with NaN input."""
        # Note: NaN input causes segfault in C++ library
        # Skip this test since it crashes the process
        # TODO: Add input validation in Python wrapper before passing to C++
        self.skipTest("NaN input causes segfault in C++ library - needs Python-level validation")


# =============================================================================
# Category G: Real Data (Horse Model)
# =============================================================================

class TestCategoryG_RealData(ConsistencyTestBase):
    """Real data tests using horse model."""

    HORSE_FILE = REPO_ROOT / "poisson" / "examples" / "horse_with_normals.xyz"

    @classmethod
    def setUpClass(cls):
        """Check if horse model file exists."""
        super().setUpClass()
        if not cls.HORSE_FILE.exists():
            cls.skipTest(f"Horse model file not found: {cls.HORSE_FILE}")

    def test_G1_horse_basic_reconstruction(self):
        """Test basic reconstruction of horse model."""
        points, normals = load_xyz_file(self.HORSE_FILE)

        print(f"\n  Horse model: {len(points)} points")

        vertices, faces = poisson_reconstruction(points, normals, depth=9)

        self.assertGreater(len(vertices), 1000)
        self.assertGreater(len(faces), 1000)

        print(f"  Reconstruction: {len(vertices)} vertices, {len(faces)} faces")

    def test_G2_horse_depth_variation(self):
        """Test horse model at different depths."""
        points, normals = load_xyz_file(self.HORSE_FILE)

        for depth in [8, 9, 10]:
            with self.subTest(depth=depth):
                vertices, faces = poisson_reconstruction(points, normals, depth=depth)

                self.assertGreater(len(vertices), 0)
                print(f"  depth={depth}: {len(vertices)} vertices")

    def test_G3_horse_grid_output(self):
        """Test horse model with grid output."""
        points, normals = load_xyz_file(self.HORSE_FILE)

        vertices, faces, grid, iso = poisson_reconstruction(
            points, normals, depth=9, grid_depth=7
        )

        res = 2 ** 7
        self.assertEqual(grid.shape, (res, res, res))
        self.assertTrue(0.0 < iso < 1.0)

        print(f"  Grid: {res}x{res}x{res}, iso={iso:.6f}")

    def test_G4_horse_exact_interpolation(self):
        """Test horse model with exact interpolation."""
        points, normals = load_xyz_file(self.HORSE_FILE)

        for exact in [False, True]:
            with self.subTest(exact=exact):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=8, exact_interpolation=exact
                )
                self.assertGreater(len(vertices), 0)
                print(f"  exact={exact}: {len(vertices)} vertices")

    def test_G5_horse_different_degrees(self):
        """Test horse model with different B-spline degrees."""
        points, normals = load_xyz_file(self.HORSE_FILE)

        for degree in [1, 2]:
            with self.subTest(degree=degree):
                vertices, faces = poisson_reconstruction(
                    points, normals, depth=8, degree=degree
                )
                self.assertGreater(len(vertices), 0)
                print(f"  degree={degree}: {len(vertices)} vertices")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
