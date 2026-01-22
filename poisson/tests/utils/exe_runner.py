import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POISSON_EXE = REPO_ROOT / "build" / "PoissonRecon"
DEFAULT_DUMP_EXE = REPO_ROOT / "build_dump" / "PoissonReconDump"

ISO_REGEX = re.compile(r"Iso-Value:\s+([0-9eE\.\+\-]+)")


def _get_exe_path(env_var: str, default_path: Path) -> Path:
    return Path(os.environ.get(env_var, str(default_path)))


def parse_iso_value(output: str):
    match = ISO_REGEX.search(output)
    if match:
        return float(match.group(1))
    return None


def parse_ply_counts(ply_path: str):
    vertex_count = 0
    face_count = 0
    with open(ply_path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line == "end_header":
                break
    return vertex_count, face_count


def run_poisson_recon(input_ply: str, output_ply: str, extra_args=None):
    exe_path = _get_exe_path("POISSON_EXE_PATH", DEFAULT_POISSON_EXE)
    args = [str(exe_path), "--in", input_ply, "--out", output_ply]
    if extra_args:
        args.extend(extra_args)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"PoissonRecon failed: {result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def run_poisson_dump(input_ply: str, output_ply: str, extra_args=None):
    exe_path = _get_exe_path("POISSON_DUMP_EXE_PATH", DEFAULT_DUMP_EXE)
    args = [str(exe_path), "--in", input_ply, "--out", output_ply]
    if extra_args:
        args.extend(extra_args)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"PoissonReconDump failed: {result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def exe_supports_btype():
    exe_path = _get_exe_path("POISSON_EXE_PATH", DEFAULT_POISSON_EXE)
    result = subprocess.run([str(exe_path)], capture_output=True, text=True)
    output = result.stdout + result.stderr
    return "--bType" in output
