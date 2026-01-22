import numpy as np
from pathlib import Path


def generate_sphere_points(n_points=1000, radius=1.0, seed=42):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= radius
    normals = points / radius
    return points.astype(np.float64), normals.astype(np.float64)


def generate_half_sphere_points(n_points=1000, radius=1.0, seed=42):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, 3))
    points[:, 2] = np.abs(points[:, 2])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= radius
    normals = points / radius
    return points.astype(np.float64), normals.astype(np.float64)


def generate_cube_points(points_per_face=200, size=1.0, seed=42):
    rng = np.random.default_rng(seed)
    half = size / 2.0
    faces = []
    normals = []
    for axis in range(3):
        for sign in [-1.0, 1.0]:
            coords = rng.uniform(-half, half, size=(points_per_face, 3))
            coords[:, axis] = sign * half
            normal = np.zeros((points_per_face, 3))
            normal[:, axis] = sign
            faces.append(coords)
            normals.append(normal)
    points = np.vstack(faces)
    normals = np.vstack(normals)
    return points.astype(np.float64), normals.astype(np.float64)


def generate_plane_points(n_points=1000, size=1.0, seed=42):
    rng = np.random.default_rng(seed)
    half = size / 2.0
    x = rng.uniform(-half, half, size=n_points)
    y = rng.uniform(-half, half, size=n_points)
    z = np.zeros_like(x)
    points = np.stack([x, y, z], axis=1)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (n_points, 1))
    return points.astype(np.float64), normals.astype(np.float64)


def load_xyz_file(path):
    path = Path(path)
    data = np.loadtxt(path, dtype=np.float64)
    points = data[:, :3]
    normals = data[:, 3:6]
    return points, normals


def save_points_ply_ascii(points, normals, filename):
    num_points = len(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "end_header\n"
    )
    with open(filename, "w", encoding="ascii") as f:
        f.write(header)
        for p, n in zip(points, normals):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n"
            )
