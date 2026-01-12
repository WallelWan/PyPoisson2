import numpy as np
import subprocess
import re
import os
import sys

# Add repo root to path so we can import the built package
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)

from poisson import poisson_reconstruction

def save_points_ply(points, normals, filename):
    num_points = len(points)
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for p, n in zip(points, normals):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

def parse_iso_value(stdout_str):
    # Look for "Iso-Value: 0.500123"
    match = re.search(r"Iso-Value:\s+([0-9\.]+)", stdout_str)
    if match:
        return float(match.group(1))
    return None

def main():
    print("Generating synthetic data (sphere).")
    n_points = 5000
    points = np.random.randn(n_points, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    normals = points.copy() # Normals for a unit sphere are the positions themselves

    input_ply = "val_input.ply"
    ref_ply = "val_ref.ply"
    
    print(f"Saving input to {input_ply}...")
    save_points_ply(points, normals, input_ply)

    print("Running Reference Executable...")
    # --verbose is needed to see Iso-Value
    cmd = [os.path.join(REPO_ROOT, "exe", "PoissonRecon"), "--in", input_ply, "--out", ref_ply, "--depth", "8", "--verbose"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running reference executable:")
        print(result.stderr)
        return

    ref_iso_value = parse_iso_value(result.stdout)
    print(f"Reference Iso-Value: {ref_iso_value}")

    print("Running pypoisson2...")
    # To get iso_value from pypoisson2, we need to request grid_depth > 0
    # Or just rely on the reconstruction result since iso_value is internal if grid is not requested.
    # But wait, we exposed iso_value only when grid_depth > 0.
    # Let's verify we get the same iso-value if we request a dummy grid.
    
    v, f, g, py_iso_value = poisson_reconstruction(points, normals, depth=8, grid_depth=1)
    print(f"Python Iso-Value:    {py_iso_value}")

    if ref_iso_value is not None:
        diff = abs(ref_iso_value - py_iso_value)
        print(f"Difference: {diff:.6e}")
        if diff < 5e-3:  # 5e-3 tolerance due to ASCII PLY precision (6 decimal places)
            print("✅ Iso-Value matches! (within ASCII PLY tolerance)")
        else:
            print("❌ Iso-Value mismatch!")

    # Compare geometry roughly (vertex count)
    # Note: Vertex count might differ slightly if float precision handling differs (text vs binary)
    # but should be very close.
    print(f"Python Mesh: {len(v)} vertices, {len(f)} faces")
    
    # Simple parse of PLY header for reference mesh
    with open(ref_ply, 'rb') as ref_file:
        content = ref_file.read()
        header_end = content.find(b"end_header")
        header = content[:header_end].decode('ascii')
        
        v_match = re.search(r"element vertex (\d+)", header)
        f_match = re.search(r"element face (\d+)", header)
        
        ref_v_count = int(v_match.group(1)) if v_match else 0
        ref_f_count = int(f_match.group(1)) if f_match else 0
        
    print(f"Ref Mesh:    {ref_v_count} vertices, {ref_f_count} faces")
    
    if abs(len(v) - ref_v_count) < 5 and abs(len(f) - ref_f_count) < 10:
         print("✅ Mesh complexity matches!")
    else:
         print("⚠️  Mesh complexity differs (could be due to IO precision)")

    # Cleanup
    if os.path.exists(input_ply): os.remove(input_ply)
    if os.path.exists(ref_ply): os.remove(ref_ply)

if __name__ == "__main__":
    main()
