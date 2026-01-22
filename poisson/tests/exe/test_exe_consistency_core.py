import os
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poisson import poisson_reconstruction
from utils.data_gen import (
    generate_sphere_points,
    generate_cube_points,
    save_points_ply_ascii,
)
from utils.exe_runner import run_poisson_recon, parse_iso_value, parse_ply_counts


class ExeConsistencyBase(unittest.TestCase):
    ISO_REL_TOL = 0.012
    ISO_ABS_TOL = 1e-4
    VERT_REL_TOL = 0.001
    FACE_REL_TOL = 0.001
    MIN_TOL = 10

    def _relative_tol(self, ref_value, rel_tol):
        return max(self.MIN_TOL, int(ref_value * rel_tol))

    def _iso_tol(self, ref_value):
        return max(self.ISO_ABS_TOL, abs(ref_value) * self.ISO_REL_TOL)

    def compare_with_exe(self, points, normals, py_kwargs, exe_args):
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f_out:
            input_ply = f_in.name
            output_ply = f_out.name

        try:
            save_points_ply_ascii(points, normals, input_ply)
            stdout = run_poisson_recon(input_ply, output_ply, exe_args)
            iso_ref = parse_iso_value(stdout)
            v_ref, f_ref = parse_ply_counts(output_ply)

            v_py, f_py, _, iso_py = poisson_reconstruction(
                points, normals, grid_depth=1, **py_kwargs
            )

            self.assertIsNotNone(iso_ref, "Iso-value not found in exe output")
            self.assertLessEqual(abs(iso_ref - iso_py), self._iso_tol(iso_ref))

            v_tol = self._relative_tol(v_ref, self.VERT_REL_TOL)
            f_tol = self._relative_tol(f_ref, self.FACE_REL_TOL)
            self.assertLessEqual(abs(v_ref - len(v_py)), v_tol)
            self.assertLessEqual(abs(f_ref - len(f_py)), f_tol)
        finally:
            for path in [input_ply, output_ply]:
                if os.path.exists(path):
                    os.remove(path)


class TestExeConsistencyCore(ExeConsistencyBase):
    def test_default_sphere(self):
        points, normals = generate_sphere_points(n_points=2000)
        self.compare_with_exe(
            points,
            normals,
            {"depth": 6},
            ["--depth", "6", "--verbose"],
        )

    def test_cube(self):
        points, normals = generate_cube_points(points_per_face=200)
        self.compare_with_exe(
            points,
            normals,
            {"depth": 6, "samples_per_node": 2.0},
            ["--depth", "6", "--samplesPerNode", "2.0", "--verbose"],
        )

    def test_scale_variation(self):
        points, normals = generate_sphere_points(n_points=1500)
        self.compare_with_exe(
            points,
            normals,
            {"depth": 6, "scale": 1.2},
            ["--depth", "6", "--scale", "1.2", "--verbose"],
        )


if __name__ == "__main__":
    unittest.main()
