import os
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poisson import poisson_reconstruction
from utils.data_gen import load_xyz_file, save_points_ply_ascii
from utils.exe_runner import run_poisson_recon, parse_iso_value, parse_ply_counts, exe_supports_btype


class TestExeBoundaryConsistency(unittest.TestCase):
    ISO_REL_TOL = 0.012
    ISO_ABS_TOL = 1e-4
    VERT_REL_TOL = 0.007
    FACE_REL_TOL = 0.007
    MIN_TOL = 15

    @classmethod
    def setUpClass(cls):
        cls.supports_btype = exe_supports_btype()

    def compare_boundary(self, boundary, btype):
        if not self.supports_btype and boundary != "neumann":
            self.skipTest("PoissonRecon built with FAST_COMPILE; boundary types not configurable")
        horse_path = REPO_ROOT / "poisson" / "examples" / "horse_with_normals.xyz"
        points, normals = load_xyz_file(horse_path)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f_out:
            input_ply = f_in.name
            output_ply = f_out.name

        try:
            save_points_ply_ascii(points, normals, input_ply)
            exe_args = ["--depth", "6", "--verbose"]
            if self.supports_btype:
                exe_args.extend(["--bType", str(btype)])
            stdout = run_poisson_recon(input_ply, output_ply, exe_args)
            iso_ref = parse_iso_value(stdout)
            v_ref, f_ref = parse_ply_counts(output_ply)

            v_py, f_py, _, iso_py = poisson_reconstruction(
                points, normals, depth=6, grid_depth=1, boundary=boundary
            )

            self.assertIsNotNone(iso_ref, "Iso-value not found in exe output")
            iso_ref_value = float(iso_ref) if iso_ref is not None else 0.0
            iso_tol = max(self.ISO_ABS_TOL, abs(iso_ref_value) * self.ISO_REL_TOL)
            self.assertLessEqual(abs(iso_ref_value - iso_py), iso_tol)

            v_tol = max(self.MIN_TOL, int(v_ref * self.VERT_REL_TOL))
            f_tol = max(self.MIN_TOL, int(f_ref * self.FACE_REL_TOL))
            self.assertLessEqual(abs(v_ref - len(v_py)), v_tol)
            self.assertLessEqual(abs(f_ref - len(f_py)), f_tol)
        finally:
            for path in [input_ply, output_ply]:
                if os.path.exists(path):
                    os.remove(path)

    def test_boundary_free(self):
        self.compare_boundary("free", 1)

    def test_boundary_dirichlet(self):
        self.compare_boundary("dirichlet", 2)

    def test_boundary_neumann(self):
        self.compare_boundary("neumann", 3)


if __name__ == "__main__":
    unittest.main()
