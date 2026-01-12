import unittest
import numpy as np
import os
import sys

# Add repo root to path so we can import the built package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poisson import poisson_reconstruction, ParallelType

class TestPoissonReconstruction(unittest.TestCase):

    def generate_sphere_points(self, n_points=1000):
        """Generates random points on a unit sphere."""
        points = np.random.randn(n_points, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        normals = points.copy()
        return points, normals

    def test_basic_sphere(self):
        """Test reconstruction of a simple sphere."""
        points, normals = self.generate_sphere_points(2000)
        
        # Run reconstruction
        # depth=6 is enough for a quick test
        vertices, faces = poisson_reconstruction(
            points, normals, 
            depth=6, 
            full_depth=4,
            verbose=True
        )
        
        print(f"Reconstructed sphere: {len(vertices)} vertices, {len(faces)} faces")
        
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)
        
        # Check if vertices are roughly on a unit sphere (radius 1.0)
        radii = np.linalg.norm(vertices, axis=1)
        mean_radius = np.mean(radii)
        print(f"Mean radius: {mean_radius}")
        
        # Allowing some margin for Poisson approximation
        self.assertAlmostEqual(mean_radius, 1.0, delta=0.2)

    def test_parallel_types(self):
        """Test different parallelization modes."""
        points, normals = self.generate_sphere_points(500)
        
        for p_type in [ParallelType.OPEN_MP, ParallelType.ASYNC, ParallelType.NONE]:
            print(f"Testing parallel_type={p_type}")
            vertices, faces = poisson_reconstruction(
                points, normals, 
                depth=5, 
                parallel_type=p_type
            )
            self.assertGreater(len(vertices), 0)

    def test_implicit_field(self):
        """Test extraction of the implicit field."""
        points, normals = self.generate_sphere_points()
        
        # Request grid output
        grid_depth = 5
        vertices, faces, grid, iso_value = poisson_reconstruction(
            points, normals, depth=6, grid_depth=grid_depth
        )
        
        # Check grid shape
        res = 2**grid_depth
        self.assertEqual(grid.shape, (res, res, res))
        
        # Check that grid is not all zeros
        self.assertFalse(np.all(grid == 0))
        
        # Check iso_value
        self.assertIsInstance(iso_value, float)
        # Iso-value is typically close to 0.5 for screened poisson
        self.assertTrue(0.0 < iso_value < 1.0, f"Iso-value {iso_value} out of expected range (0, 1)")
        
        print(f"Grid shape: {grid.shape}, Mean value: {np.mean(grid)}, Iso-value: {iso_value}")

    def test_invalid_points_shape(self):
        """Test points shape validation."""
        points = np.zeros((10,), dtype=np.float64)
        normals = np.zeros((10, 3), dtype=np.float64)

        with self.assertRaises(ValueError):
            poisson_reconstruction(points, normals)

    def test_invalid_normals_shape(self):
        """Test normals shape validation."""
        points = np.zeros((10, 3), dtype=np.float64)
        normals = np.zeros((10, 2), dtype=np.float64)

        with self.assertRaises(ValueError):
            poisson_reconstruction(points, normals)

    def test_empty_input_with_outputs(self):
        """Test empty input with grid and advanced outputs."""
        points = np.zeros((0, 3), dtype=np.float32)
        normals = np.zeros((0, 3), dtype=np.float32)

        vertices, faces, grid, iso_value, densities, gradients = poisson_reconstruction(
            points,
            normals,
            grid_depth=4,
            output_density=True,
            output_gradients=True,
        )

        self.assertEqual(vertices.shape, (0, 3))
        self.assertEqual(faces.shape, (0, 3))
        self.assertEqual(grid.shape, (0, 0, 0))
        self.assertEqual(iso_value, 0.0)
        self.assertEqual(densities.shape, (0,))
        self.assertEqual(gradients.shape, (0, 3))

    def test_validate_finite(self):
        """Test finite validation rejects NaN/Inf."""
        points, normals = self.generate_sphere_points(10)

        points[0, 0] = np.nan
        with self.assertRaises(ValueError):
            poisson_reconstruction(points, normals, validate_finite=True)

        points, normals = self.generate_sphere_points(10)
        normals[0, 1] = np.inf
        with self.assertRaises(ValueError):
            poisson_reconstruction(points, normals, validate_finite=True)

    def test_force_big(self):
        """Test forcing big-index library."""
        points, normals = self.generate_sphere_points(200)
        vertices, faces = poisson_reconstruction(points, normals, depth=5, force_big=True)
        self.assertGreater(len(vertices), 0)
        self.assertEqual(faces.dtype, np.int64)

if __name__ == '__main__':
    unittest.main()
