import unittest
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(Path(__file__).parent))

from poisson import poisson_reconstruction
from test_data_generator import generate_sphere_points

class TestPhase2Boundary(unittest.TestCase):
    def setUp(self):
        self.points, self.normals = generate_sphere_points(n_points=2000, radius=1.0)
        # Add some noise to make reconstruction non-trivial
        # self.points += np.random.normal(0, 0.01, self.points.shape)

    def test_boundary_types(self):
        """Test all three boundary types to ensure they run without crashing."""
        boundaries = ['neumann', 'dirichlet', 'free']
        results = {}

        for b in boundaries:
            print(f"Testing boundary: {b}")
            verts, faces = poisson_reconstruction(
                self.points, self.normals, 
                depth=6, 
                boundary=b,
                verbose=True
            )
            results[b] = (len(verts), len(faces))
            self.assertTrue(len(verts) > 0)
            self.assertTrue(len(faces) > 0)
        
        print("Boundary results (V, F):", results)
        
        # Heuristic check: Different boundaries should produce slightly different meshes
        # though on a closed sphere, differences might be minimal unless the field extends to the bounding box.
        # Neumann is default.
        
        # To make differences more visible, we might need an open surface or low depth relative to bounding box?
        # But for now, just ensure they run and produce valid output.

    def test_dirichlet_erode(self):
        """Test that dirichlet_erode changes the output (iso-value or mesh)."""
        # Baseline: Dirichlet without erode
        # Request grid_depth to get iso_value
        v1, f1, g1, iso1 = poisson_reconstruction(
            self.points, self.normals,
            depth=6,
            boundary='dirichlet',
            dirichlet_erode=False,
            grid_depth=4
        )

        # Test: Dirichlet with erode
        v2, f2, g2, iso2 = poisson_reconstruction(
            self.points, self.normals,
            depth=6,
            boundary='dirichlet',
            dirichlet_erode=True,
            grid_depth=4
        )

        print(f"Dirichlet No-Erode: Iso={iso1}, V={len(v1)}")
        print(f"Dirichlet Erode   : Iso={iso2}, V={len(v2)}")

        # Erode should affect the field, hence the iso-value used for extraction might change
        # or at least the underlying field values near boundary.
        self.assertNotEqual(iso1, iso2)

    def test_invalid_boundary(self):
        """Test that invalid boundary string raises ValueError."""
        with self.assertRaises(ValueError):
            poisson_reconstruction(self.points, self.normals, boundary='invalid_option')

    def test_boundary_case_insensitive(self):
        """Test that boundary parsing is case-insensitive."""
        vertices, faces = poisson_reconstruction(
            self.points, self.normals, depth=5, boundary='NeUmAnN'
        )
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)

    def test_boundary_empty_string(self):
        """Test that empty boundary string raises ValueError."""
        with self.assertRaises(ValueError):
            poisson_reconstruction(self.points, self.normals, boundary='')


if __name__ == '__main__':
    unittest.main()
