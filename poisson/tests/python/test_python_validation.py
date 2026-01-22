import unittest
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from poisson import poisson_reconstruction


class TestPythonValidation(unittest.TestCase):
    def test_invalid_points_shape(self):
        points = np.zeros((10,), dtype=np.float64)
        normals = np.zeros((10, 3), dtype=np.float64)
        with self.assertRaises(ValueError):
            poisson_reconstruction(points, normals)

    def test_invalid_normals_shape(self):
        points = np.zeros((10, 3), dtype=np.float64)
        normals = np.zeros((10, 2), dtype=np.float64)
        with self.assertRaises(ValueError):
            poisson_reconstruction(points, normals)

    def test_validate_finite(self):
        points = np.zeros((10, 3), dtype=np.float64)
        normals = np.zeros((10, 3), dtype=np.float64)
        points[0, 0] = np.nan
        with self.assertRaises(ValueError):
            poisson_reconstruction(points, normals, validate_finite=True)

        points = np.zeros((10, 3), dtype=np.float64)
        normals = np.zeros((10, 3), dtype=np.float64)
        normals[0, 1] = np.inf
        with self.assertRaises(ValueError):
            poisson_reconstruction(points, normals, validate_finite=True)

    def test_empty_input_outputs(self):
        points = np.zeros((0, 3), dtype=np.float64)
        normals = np.zeros((0, 3), dtype=np.float64)
        vertices, faces, grid, iso_value = poisson_reconstruction(
            points, normals, grid_depth=4
        )
        self.assertEqual(vertices.shape, (0, 3))
        self.assertEqual(faces.shape, (0, 3))
        self.assertEqual(grid.shape, (0, 0, 0))
        self.assertEqual(iso_value, 0.0)


if __name__ == "__main__":
    unittest.main()
