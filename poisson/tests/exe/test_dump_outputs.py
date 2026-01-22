import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poisson import poisson_reconstruction, ParallelType
from utils.data_gen import generate_sphere_points, save_points_ply_ascii
from utils.dump_reader import read_density_file, read_gradient_file
from utils.exe_runner import run_poisson_dump


class TestDumpOutputConsistency(unittest.TestCase):
    DENSITY_TOL = 1e-6
    GRADIENT_TOL = 1e-2

    def test_density_and_gradients(self):
        points, normals = generate_sphere_points(n_points=2000)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f_out, \
             tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f_density, \
             tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f_gradients:
            input_ply = f_in.name
            output_ply = f_out.name
            density_path = f_density.name
            gradients_path = f_gradients.name

        try:
            save_points_ply_ascii(points, normals, input_ply)
            run_poisson_dump(
                input_ply,
                output_ply,
                [
                    "--depth", "6",
                    "--density",
                    "--gradients",
                    "--parallel", "2",
                    "--densityOut", density_path,
                    "--gradientsOut", gradients_path,
                ],
            )

            densities_dump = read_density_file(density_path)
            gradients_dump = read_gradient_file(gradients_path)

            v_py, f_py, densities_py, gradients_py = poisson_reconstruction(
                points,
                normals,
                depth=6,
                parallel_type=ParallelType.NONE,
                output_density=True,
                output_gradients=True,
            )

            self.assertEqual(len(densities_dump), len(densities_py))
            self.assertEqual(gradients_dump.shape, gradients_py.shape)
            self.assertTrue(np.allclose(densities_dump, densities_py, atol=self.DENSITY_TOL))
            self.assertTrue(np.allclose(gradients_dump, gradients_py, atol=self.GRADIENT_TOL))
        finally:
            for path in [input_ply, output_ply, density_path, gradients_path]:
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()
