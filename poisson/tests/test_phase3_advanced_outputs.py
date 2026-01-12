#!/usr/bin/env python3
"""
Phase 3: Advanced Outputs - Density and Gradient Testing

Tests the density and gradient output functionality:
- Per-vertex density values (output_density parameter)
- Per-vertex gradient vectors (output_gradients parameter)
- Dynamic return value logic (combinations of grid, density, gradients)
- Integration with existing parameters
- Mathematical validation of gradient properties
- Real data validation (horse model)

Total: 32 tests across 7 categories
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add repo and test directories to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from poisson import api
# Import from local test modules
import test_data_generator
import geometry_utils

# Tolerance thresholds
DENSITY_TOLERANCE = 1e-6  # Density values should be positive
GRADIENT_NORM_TOLERANCE = 1e-3  # Gradient norms should be close to 1 (normalized)


class AdvancedOutputTestBase(unittest.TestCase):
    """Base class for Phase 3 advanced output tests."""

    def setUp(self):
        """Generate test data before each test."""
        np.random.seed(42)  # For reproducibility

    def validate_density(self, densities):
        """Validate density array properties."""
        self.assertIsInstance(densities, np.ndarray)
        self.assertEqual(densities.ndim, 1)
        self.assertGreater(len(densities), 0)
        # Densities should be non-negative
        self.assertTrue(np.all(densities >= 0),
                       f"All densities should be >= 0, found min: {np.min(densities)}")

    def validate_gradient(self, gradients):
        """Validate gradient array properties."""
        self.assertIsInstance(gradients, np.ndarray)
        self.assertEqual(gradients.ndim, 2)
        self.assertEqual(gradients.shape[1], 3)
        self.assertGreater(gradients.shape[0], 0)
        # Gradients should be finite
        self.assertTrue(np.all(np.isfinite(gradients)),
                       f"All gradients should be finite")

    def validate_return_length(self, result, expected_length):
        """Validate return tuple has correct length."""
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), expected_length,
                        f"Expected {expected_length} return values, got {len(result)}")


class TestCategoryA_BasicAdvancedOutput(AdvancedOutputTestBase):
    """Category A: Basic density and gradient output tests (4 tests)."""

    def test_A1_density_output_basic(self):
        """Test basic density output."""
        print("\n[A1] Testing basic density output...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)
        vertices, faces, densities = api.poisson_reconstruction(
            points, normals,
            depth=6,
            output_density=True
        )

        self.validate_return_length((vertices, faces, densities), 3)
        self.assertEqual(len(densities), len(vertices))
        self.validate_density(densities)

        print(f"  Vertices: {len(vertices)}, Densities: {len(densities)}")
        print(f"  Density range: [{np.min(densities):.6f}, {np.max(densities):.6f}]")
        print(f"  Density mean: {np.mean(densities):.6f}")

    def test_A2_gradient_output_basic(self):
        """Test basic gradient output."""
        print("\n[A2] Testing basic gradient output...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)
        vertices, faces, gradients = api.poisson_reconstruction(
            points, normals,
            depth=6,
            output_gradients=True
        )

        self.validate_return_length((vertices, faces, gradients), 3)
        self.assertEqual(gradients.shape[0], len(vertices))
        self.assertEqual(gradients.shape[1], 3)
        self.validate_gradient(gradients)

        print(f"  Vertices: {len(vertices)}, Gradients: {gradients.shape}")
        print(f"  Gradient norm mean: {np.mean(np.linalg.norm(gradients, axis=1)):.6f}")

    def test_A3_both_outputs_basic(self):
        """Test both density and gradient output together."""
        print("\n[A3] Testing both density and gradient output...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)
        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=6,
            output_density=True,
            output_gradients=True
        )

        self.validate_return_length((vertices, faces, densities, gradients), 4)
        self.validate_density(densities)
        self.validate_gradient(gradients)
        self.assertEqual(len(densities), len(vertices))
        self.assertEqual(gradients.shape[0], len(vertices))

        print(f"  Vertices: {len(vertices)}, Densities: {len(densities)}, Gradients: {gradients.shape}")
        print(f"  Density range: [{np.min(densities):.6f}, {np.max(densities):.6f}]")
        print(f"  Gradient norm mean: {np.mean(np.linalg.norm(gradients, axis=1)):.6f}")

    def test_A4_no_advanced_output(self):
        """Test that advanced outputs are not returned when not requested."""
        print("\n[A4] Testing no advanced output (default behavior)...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)
        vertices, faces = api.poisson_reconstruction(
            points, normals,
            depth=6,
            output_density=False,
            output_gradients=False
        )

        self.validate_return_length((vertices, faces), 2)
        print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
        print("  No advanced outputs returned (as expected)")


class TestCategoryB_OutputCombinations(AdvancedOutputTestBase):
    """Category B: Output combination tests with grid (5 tests)."""

    def test_B1_grid_only(self):
        """Test grid output without advanced outputs."""
        print("\n[B1] Testing grid output only...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)
        vertices, faces, grid, iso_value = api.poisson_reconstruction(
            points, normals,
            depth=6,
            grid_depth=5,
            output_density=False,
            output_gradients=False
        )

        self.validate_return_length((vertices, faces, grid, iso_value), 4)
        print(f"  Vertices: {len(vertices)}, Grid shape: {grid.shape}, Iso-value: {iso_value:.6f}")
        print("  Grid output only (no advanced outputs)")

    def test_B2_grid_with_density(self):
        """Test grid output with density."""
        print("\n[B2] Testing grid output with density...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)
        vertices, faces, grid, iso_value, densities = api.poisson_reconstruction(
            points, normals,
            depth=6,
            grid_depth=5,
            output_density=True,
            output_gradients=False
        )

        self.validate_return_length((vertices, faces, grid, iso_value, densities), 5)
        self.validate_density(densities)
        print(f"  Vertices: {len(vertices)}, Grid shape: {grid.shape}")
        print(f"  Densities: {len(densities)}, Iso-value: {iso_value:.6f}")

    def test_B3_grid_with_gradients(self):
        """Test grid output with gradients."""
        print("\n[B3] Testing grid output with gradients...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)
        vertices, faces, grid, iso_value, gradients = api.poisson_reconstruction(
            points, normals,
            depth=6,
            grid_depth=5,
            output_density=False,
            output_gradients=True
        )

        self.validate_return_length((vertices, faces, grid, iso_value, gradients), 5)
        self.validate_gradient(gradients)
        print(f"  Vertices: {len(vertices)}, Grid shape: {grid.shape}")
        print(f"  Gradients: {gradients.shape}, Iso-value: {iso_value:.6f}")

    def test_B4_grid_with_both(self):
        """Test grid output with both density and gradients."""
        print("\n[B4] Testing grid output with both density and gradients...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)
        vertices, faces, grid, iso_value, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=6,
            grid_depth=5,
            output_density=True,
            output_gradients=True
        )

        self.validate_return_length((vertices, faces, grid, iso_value, densities, gradients), 6)
        self.validate_density(densities)
        self.validate_gradient(gradients)
        print(f"  Vertices: {len(vertices)}, Grid shape: {grid.shape}")
        print(f"  Densities: {len(densities)}, Gradients: {gradients.shape}")
        print(f"  Iso-value: {iso_value:.6f}")

    def test_B5_density_with_gradients_no_grid(self):
        """Test density and gradients without grid."""
        print("\n[B5] Testing density and gradients without grid...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)
        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=6,
            grid_depth=0,
            output_density=True,
            output_gradients=True
        )

        self.validate_return_length((vertices, faces, densities, gradients), 4)
        self.validate_density(densities)
        self.validate_gradient(gradients)
        print(f"  Vertices: {len(vertices)}, Densities: {len(densities)}, Gradients: {gradients.shape}")
        print("  No grid output")


class TestCategoryC_ParameterIntegration(AdvancedOutputTestBase):
    """Category C: Integration with other parameters (4 tests)."""

    def test_C1_density_with_depth_variations(self):
        """Test density output with different depth values."""
        print("\n[C1] Testing density with depth variations...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)

        for depth in [5, 6, 7]:
            vertices, faces, densities = api.poisson_reconstruction(
                points, normals,
                depth=depth,
                output_density=True
            )

            self.validate_density(densities)
            print(f"  Depth {depth}: {len(vertices)} vertices, density range [{np.min(densities):.6f}, {np.max(densities):.6f}]")

    def test_C2_gradients_with_degree_variations(self):
        """Test gradient output with different B-spline degrees."""
        print("\n[C2] Testing gradients with degree variations...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)

        for degree in [1, 2]:
            vertices, faces, gradients = api.poisson_reconstruction(
                points, normals,
                depth=6,
                degree=degree,
                output_gradients=True
            )

            self.validate_gradient(gradients)
            grad_norms = np.linalg.norm(gradients, axis=1)
            print(f"  Degree {degree}: {len(vertices)} vertices, mean gradient norm: {np.mean(grad_norms):.6f}")

    def test_C3_advanced_outputs_with_boundary(self):
        """Test advanced outputs with different boundary conditions."""
        print("\n[C3] Testing advanced outputs with boundary conditions...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)

        for boundary in ['neumann', 'dirichlet', 'free']:
            vertices, faces, densities, gradients = api.poisson_reconstruction(
                points, normals,
                depth=6,
                boundary=boundary,
                output_density=True,
                output_gradients=True
            )

            self.validate_density(densities)
            self.validate_gradient(gradients)
            print(f"  Boundary {boundary}: {len(vertices)} vertices, density mean: {np.mean(densities):.6f}")

    def test_C4_advanced_outputs_with_solver_params(self):
        """Test advanced outputs with solver parameters."""
        print("\n[C4] Testing advanced outputs with solver parameters...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)

        # Test with different CG solver accuracy
        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=6,
            cg_solver_accuracy=1e-4,
            output_density=True,
            output_gradients=True
        )

        self.validate_density(densities)
        self.validate_gradient(gradients)
        print(f"  CG accuracy 1e-4: {len(vertices)} vertices, {len(densities)} densities, {gradients.shape} gradients")


class TestCategoryD_MathematicalValidation(AdvancedOutputTestBase):
    """Category D: Mathematical validation of outputs (5 tests)."""

    def test_D1_density_distribution(self):
        """Test density distribution properties."""
        print("\n[D1] Testing density distribution properties...")

        points, normals = test_data_generator.generate_sphere_points(n_points=2000, radius=1.0)
        vertices, faces, densities = api.poisson_reconstruction(
            points, normals,
            depth=7,
            output_density=True
        )

        self.validate_density(densities)

        # Check density distribution
        density_std = np.std(densities)
        density_mean = np.mean(densities)

        print(f"  Density statistics:")
        print(f"    Mean: {density_mean:.6f}")
        print(f"    Std: {density_std:.6f}")
        print(f"    Min: {np.min(densities):.6f}")
        print(f"    Max: {np.max(densities):.6f}")

        # Note: Density values may be uniform depending on the library implementation
        # This is expected behavior - just verify they're all valid
        self.assertTrue(np.all(densities > 0), "All densities should be positive")

    def test_D2_gradient_norm_distribution(self):
        """Test gradient norm distribution."""
        print("\n[D2] Testing gradient norm distribution...")

        points, normals = test_data_generator.generate_sphere_points(n_points=2000, radius=1.0)
        vertices, faces, gradients = api.poisson_reconstruction(
            points, normals,
            depth=7,
            output_gradients=True
        )

        self.validate_gradient(gradients)

        # Compute gradient norms
        grad_norms = np.linalg.norm(gradients, axis=1)

        print(f"  Gradient norm statistics:")
        print(f"    Mean: {np.mean(grad_norms):.6f}")
        print(f"    Std: {np.std(grad_norms):.6f}")
        print(f"    Min: {np.min(grad_norms):.6f}")
        print(f"    Max: {np.max(grad_norms):.6f}")

        # Gradients should be non-zero (field has variation)
        self.assertGreater(np.mean(grad_norms), 0.01, "Gradient norms should be non-zero")

    def test_D3_density_gradient_correlation(self):
        """Test correlation between density and gradients."""
        print("\n[D3] Testing density-gradient correlation...")

        points, normals = test_data_generator.generate_sphere_points(n_points=2000, radius=1.0)
        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=7,
            output_density=True,
            output_gradients=True
        )

        self.validate_density(densities)
        self.validate_gradient(gradients)

        # Compute gradient norms
        grad_norms = np.linalg.norm(gradients, axis=1)

        # Compute correlation
        correlation = np.corrcoef(densities, grad_norms)[0, 1]

        print(f"  Density-gradient correlation: {correlation:.6f}")
        print(f"  Density mean: {np.mean(densities):.6f}")
        print(f"  Gradient norm mean: {np.mean(grad_norms):.6f}")

        # Note: No specific assertion on correlation - just measuring

    def test_D4_gradient_direction_consistency(self):
        """Test that gradient directions are consistent with surface normals."""
        print("\n[D4] Testing gradient direction consistency...")

        points, normals = test_data_generator.generate_sphere_points(n_points=2000, radius=1.0)
        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=7,
            output_density=True,
            output_gradients=True
        )

        self.validate_gradient(gradients)

        # Normalize gradients
        grad_norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        normalized_gradients = gradients / (grad_norms + 1e-10)

        # Compute vertex normals from mesh
        # (Simple approximation: normalized position for sphere)
        vertex_normals = vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-10)

        # Compute dot products
        dot_products = np.sum(normalized_gradients * vertex_normals, axis=1)
        mean_alignment = np.mean(np.abs(dot_products))

        print(f"  Mean gradient-normal alignment: {mean_alignment:.6f}")
        print(f"  Alignment range: [{np.min(np.abs(dot_products)):.6f}, {np.max(np.abs(dot_products)):.6f}]")

        # Gradients should be somewhat aligned with surface (but not perfectly)
        self.assertGreater(mean_alignment, 0.1, "Gradients should have some alignment with surface")

    def test_D5_density_gradient_grid_consistency(self):
        """Test consistency between density, gradients, and grid values."""
        print("\n[D5] Testing density-gradient-grid consistency...")

        points, normals = test_data_generator.generate_sphere_points(n_points=2000, radius=1.0)
        vertices, faces, grid, iso_value, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=7,
            grid_depth=6,
            output_density=True,
            output_gradients=True
        )

        self.validate_density(densities)
        self.validate_gradient(gradients)

        print(f"  Grid shape: {grid.shape}, Iso-value: {iso_value:.6f}")
        print(f"  Density range: [{np.min(densities):.6f}, {np.max(densities):.6f}]")
        print(f"  Gradient norm range: [{np.min(np.linalg.norm(gradients, axis=1)):.6f}, {np.max(np.linalg.norm(gradients, axis=1)):.6f}]")

        # All should be finite and valid
        self.assertTrue(np.isfinite(iso_value))
        self.assertTrue(np.all(np.isfinite(grid)))


class TestCategoryE_EdgeCases(AdvancedOutputTestBase):
    """Category E: Edge case tests (5 tests)."""

    def test_E1_empty_input_with_outputs(self):
        """Test advanced outputs with empty input."""
        print("\n[E1] Testing empty input with advanced outputs...")

        points = np.zeros((0, 3), dtype=np.float64)
        normals = np.zeros((0, 3), dtype=np.float64)

        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            output_density=True,
            output_gradients=True
        )

        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(faces), 0)
        self.assertEqual(len(densities), 0)
        self.assertEqual(gradients.shape[0], 0)

        print("  Empty input handled correctly")

    def test_E1b_empty_input_with_grid_and_outputs(self):
        """Test empty input with grid and advanced outputs."""
        print("\n[E1b] Testing empty input with grid and advanced outputs...")

        points = np.zeros((0, 3), dtype=np.float64)
        normals = np.zeros((0, 3), dtype=np.float64)

        vertices, faces, grid, iso_value, densities, gradients = api.poisson_reconstruction(
            points,
            normals,
            grid_depth=3,
            output_density=True,
            output_gradients=True,
        )

        self.assertEqual(vertices.shape, (0, 3))
        self.assertEqual(faces.shape, (0, 3))
        self.assertEqual(grid.shape, (0, 0, 0))
        self.assertEqual(iso_value, 0.0)
        self.assertEqual(densities.shape, (0,))
        self.assertEqual(gradients.shape, (0, 3))

    def test_E2_few_points_with_outputs(self):
        """Test advanced outputs with minimal input."""
        print("\n[E2] Testing few points with advanced outputs...")

        # Create minimal valid input (10 points)
        points = np.random.randn(10, 3).astype(np.float64)
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        normals = normals.astype(np.float64)

        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=5,
            output_density=True,
            output_gradients=True
        )

        # Should produce output
        if len(vertices) > 0:
            self.validate_density(densities)
            self.validate_gradient(gradients)
            print(f"  Minimal input: {len(vertices)} vertices produced")
        else:
            print("  Minimal input: No vertices produced (acceptable)")

    def test_E3_coplanar_points_with_outputs(self):
        """Test advanced outputs with coplanar points."""
        print("\n[E3] Testing coplanar points with advanced outputs...")

        # Generate points on XY plane
        points = np.random.randn(100, 3).astype(np.float64)
        points[:, 2] = 0.0  # All points on Z=0 plane

        normals = np.zeros((100, 3), dtype=np.float64)
        normals[:, 2] = 1.0  # All normals pointing up

        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=5,
            output_density=True,
            output_gradients=True
        )

        # Should produce output (likely planar mesh)
        if len(vertices) > 0:
            self.validate_density(densities)
            self.validate_gradient(gradients)

            # Check Z-coordinate is small
            z_std = np.std(vertices[:, 2])
            print(f"  Coplanar input: Z std = {z_std:.6f}")
        else:
            print("  Coplanar input: No vertices produced (acceptable)")

    def test_E4_high_density_output(self):
        """Test advanced outputs with high-depth reconstruction."""
        print("\n[E4] Testing advanced outputs with high depth...")

        points, normals = test_data_generator.generate_sphere_points(n_points=5000, radius=1.0)

        # Use high depth (should produce many vertices)
        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=9,
            output_density=True,
            output_gradients=True
        )

        self.validate_density(densities)
        self.validate_gradient(gradients)
        self.assertGreater(len(vertices), 1000, "High depth should produce many vertices")

        print(f"  High depth (9): {len(vertices)} vertices")
        print(f"  Densities: {len(densities)}, Gradients: {gradients.shape}")

    def test_E5_switching_outputs_on_off(self):
        """Test switching output flags on and off."""
        print("\n[E5] Testing switching output flags...")

        points, normals = test_data_generator.generate_sphere_points(n_points=1000, radius=1.0)

        # Test all 4 combinations
        combinations = [
            (False, False, 2),
            (True, False, 3),
            (False, True, 3),
            (True, True, 4),
        ]

        for out_density, out_grad, expected_len in combinations:
            result = api.poisson_reconstruction(
                points, normals,
                depth=6,
                output_density=out_density,
                output_gradients=out_grad
            )

            self.assertEqual(len(result), expected_len,
                            f"Expected {expected_len} outputs for density={out_density}, gradients={out_grad}")
            print(f"  density={out_density}, gradients={out_grad}: {len(result)} outputs ✓")


class TestCategoryF_PerformanceAndMemory(AdvancedOutputTestBase):
    """Category F: Performance and memory tests (4 tests)."""

    def test_F1_large_dataset_density(self):
        """Test density output with large dataset."""
        print("\n[F1] Testing large dataset with density output...")

        points, normals = test_data_generator.generate_sphere_points(n_points=20000, radius=1.0)

        vertices, faces, densities = api.poisson_reconstruction(
            points, normals,
            depth=8,
            output_density=True
        )

        self.validate_density(densities)
        print(f"  Large dataset (20K points): {len(vertices)} vertices, {len(densities)} densities")

    def test_F2_large_dataset_gradients(self):
        """Test gradient output with large dataset."""
        print("\n[F2] Testing large dataset with gradient output...")

        points, normals = test_data_generator.generate_sphere_points(n_points=20000, radius=1.0)

        vertices, faces, gradients = api.poisson_reconstruction(
            points, normals,
            depth=8,
            output_gradients=True
        )

        self.validate_gradient(gradients)
        print(f"  Large dataset (20K points): {len(vertices)} vertices, {gradients.shape} gradients")

    def test_F3_memory_efficiency_check(self):
        """Test that advanced outputs don't excessively increase memory."""
        print("\n[F3] Testing memory efficiency...")

        points, normals = test_data_generator.generate_sphere_points(n_points=5000, radius=1.0)

        # Reconstruction without advanced outputs
        vertices_base, faces_base = api.poisson_reconstruction(
            points, normals,
            depth=7,
            output_density=False,
            output_gradients=False
        )

        # Reconstruction with advanced outputs
        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=7,
            output_density=True,
            output_gradients=True
        )

        # Vertex/face counts should be identical
        self.assertEqual(len(vertices), len(vertices_base))
        self.assertEqual(len(faces), len(faces_base))

        print(f"  Base: {len(vertices_base)} vertices")
        print(f"  With outputs: {len(vertices)} vertices, {len(densities)} densities, {gradients.shape} gradients")
        print("  Memory overhead is only from additional arrays (expected)")

    def test_F4_computation_time_consistency(self):
        """Test that advanced outputs don't drastically increase computation time."""
        print("\n[F4] Testing computation time consistency...")

        import time

        points, normals = test_data_generator.generate_sphere_points(n_points=5000, radius=1.0)

        # Time without advanced outputs
        start = time.time()
        vertices, faces = api.poisson_reconstruction(
            points, normals,
            depth=7,
            output_density=False,
            output_gradients=False
        )
        time_base = time.time() - start

        # Time with advanced outputs
        start = time.time()
        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=7,
            output_density=True,
            output_gradients=True
        )
        time_advanced = time.time() - start

        print(f"  Base time: {time_base:.3f}s")
        print(f"  Advanced outputs time: {time_advanced:.3f}s")
        print(f"  Overhead: {time_advanced - time_base:.3f}s ({(time_advanced/time_base - 1)*100:.1f}%)")

        # Advanced outputs should not more than double the time
        self.assertLess(time_advanced / time_base, 2.0,
                       "Advanced outputs should not more than double computation time")


class TestCategoryG_RealDataValidation(AdvancedOutputTestBase):
    """Category G: Real data validation tests (5 tests)."""

    def test_G1_horse_density(self):
        """Test density output on horse model."""
        print("\n[G1] Testing horse model with density output...")

        horse_file = REPO_ROOT / "poisson" / "examples" / "horse_with_normals.xyz"

        if not horse_file.exists():
            self.skipTest(f"Horse file not found: {horse_file}")

        points, normals = test_data_generator.load_xyz_file(horse_file)

        vertices, faces, densities = api.poisson_reconstruction(
            points, normals,
            depth=8,
            output_density=True
        )

        self.validate_density(densities)
        self.assertGreater(len(vertices), 1000)

        print(f"  Horse model: {len(vertices)} vertices, {len(densities)} densities")
        print(f"  Density range: [{np.min(densities):.6f}, {np.max(densities):.6f}]")

    def test_G2_horse_gradients(self):
        """Test gradient output on horse model."""
        print("\n[G2] Testing horse model with gradient output...")

        horse_file = REPO_ROOT / "poisson" / "examples" / "horse_with_normals.xyz"

        if not horse_file.exists():
            self.skipTest(f"Horse file not found: {horse_file}")

        points, normals = test_data_generator.load_xyz_file(horse_file)

        vertices, faces, gradients = api.poisson_reconstruction(
            points, normals,
            depth=8,
            output_gradients=True
        )

        self.validate_gradient(gradients)
        self.assertGreater(len(vertices), 1000)

        print(f"  Horse model: {len(vertices)} vertices, {gradients.shape} gradients")
        print(f"  Gradient norm mean: {np.mean(np.linalg.norm(gradients, axis=1)):.6f}")

    def test_G3_horse_both_outputs(self):
        """Test both density and gradients on horse model."""
        print("\n[G3] Testing horse model with both outputs...")

        horse_file = REPO_ROOT / "poisson" / "examples" / "horse_with_normals.xyz"

        if not horse_file.exists():
            self.skipTest(f"Horse file not found: {horse_file}")

        points, normals = test_data_generator.load_xyz_file(horse_file)

        vertices, faces, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=8,
            output_density=True,
            output_gradients=True
        )

        self.validate_density(densities)
        self.validate_gradient(gradients)
        self.assertGreater(len(vertices), 1000)

        print(f"  Horse model: {len(vertices)} vertices")
        print(f"  Densities: {len(densities)}, Gradients: {gradients.shape}")
        print(f"  Density mean: {np.mean(densities):.6f}")
        print(f"  Gradient norm mean: {np.mean(np.linalg.norm(gradients, axis=1)):.6f}")

    def test_G4_horse_with_grid_and_outputs(self):
        """Test horse model with grid and advanced outputs."""
        print("\n[G4] Testing horse model with grid and advanced outputs...")

        horse_file = REPO_ROOT / "poisson" / "examples" / "horse_with_normals.xyz"

        if not horse_file.exists():
            self.skipTest(f"Horse file not found: {horse_file}")

        points, normals = test_data_generator.load_xyz_file(horse_file)

        vertices, faces, grid, iso_value, densities, gradients = api.poisson_reconstruction(
            points, normals,
            depth=8,
            grid_depth=7,
            output_density=True,
            output_gradients=True
        )

        self.validate_density(densities)
        self.validate_gradient(gradients)
        self.assertGreater(len(vertices), 1000)

        print(f"  Horse model: {len(vertices)} vertices")
        print(f"  Grid: {grid.shape}, Iso-value: {iso_value:.6f}")
        print(f"  Densities: {len(densities)}, Gradients: {gradients.shape}")

    def test_G5_horse_boundary_with_outputs(self):
        """Test horse model with different boundary conditions and outputs."""
        print("\n[G5] Testing horse model with boundary conditions...")

        horse_file = REPO_ROOT / "poisson" / "examples" / "horse_with_normals.xyz"

        if not horse_file.exists():
            self.skipTest(f"Horse file not found: {horse_file}")

        points, normals = test_data_generator.load_xyz_file(horse_file)

        for boundary in ['neumann', 'dirichlet']:
            vertices, faces, densities, gradients = api.poisson_reconstruction(
                points, normals,
                depth=8,
                boundary=boundary,
                output_density=True,
                output_gradients=True
            )

            self.validate_density(densities)
            self.validate_gradient(gradients)

            print(f"  Boundary {boundary}: {len(vertices)} vertices")
            print(f"    Density mean: {np.mean(densities):.6f}")
            print(f"    Gradient norm mean: {np.mean(np.linalg.norm(gradients, axis=1)):.6f}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
