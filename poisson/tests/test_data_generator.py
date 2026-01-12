"""
Test data generation utilities for pypoisson2 consistency testing.

Provides functions to generate synthetic point clouds and load real-world data
in various formats (XYZ, PLY).
"""

import numpy as np
from pathlib import Path
import struct


def generate_sphere_points(n_points=1000, radius=1.0, noise=0.0):
    """
    Generate random points on a sphere surface.

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    radius : float
        Sphere radius (default 1.0).
    noise : float
        Gaussian noise standard deviation (default 0.0).

    Returns
    -------
    points : (N, 3) array
        Point positions.
    normals : (N, 3) array
        Point normals (pointing outward).
    """
    # Generate random points using spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.stack([x, y, z], axis=1)

    # Add noise if requested
    if noise > 0:
        points += np.random.randn(n_points, 3) * noise

    # Normals for a sphere are the normalized positions
    normals = points.copy()
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return points, normals


def generate_cube_points(n_points_per_face=200, size=2.0, noise=0.0):
    """
    Generate random points on a cube surface.

    Parameters
    ----------
    n_points_per_face : int
        Number of points per face (6 faces total).
    size : float
        Cube size (default 2.0, ranging from -size/2 to size/2).
    noise : float
        Gaussian noise standard deviation (default 0.0).

    Returns
    -------
    points : (N, 3) array
        Point positions.
    normals : (N, 3) array
        Point normals.
    """
    half_size = size / 2
    points_list = []
    normals_list = []

    # Generate points for each of the 6 faces
    for axis in range(3):
        for sign in [-1, 1]:
            # Fixed coordinate for this face
            fixed = sign * half_size

            # Varying coordinates
            a = np.random.uniform(-half_size, half_size, n_points_per_face)
            b = np.random.uniform(-half_size, half_size, n_points_per_face)

            face_points = np.zeros((n_points_per_face, 3))

            # Set fixed coordinate and varying coordinates
            face_points[:, axis] = fixed
            face_points[:, (axis + 1) % 3] = a
            face_points[:, (axis + 2) % 3] = b

            # Normal points outward
            normal = np.zeros(3)
            normal[axis] = sign

            points_list.append(face_points)
            normals_list.append(np.tile(normal, (n_points_per_face, 1)))

    points = np.vstack(points_list)
    normals = np.vstack(normals_list)

    # Add noise if requested (noise added along normal direction)
    if noise > 0:
        points += normals * np.random.randn(len(points), 1) * noise

    return points, normals


def generate_plane_points(n_points=1000, noise=0.01):
    """
    Generate random points on a plane (XY plane).

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    noise : float
        Gaussian noise standard deviation (default 0.01).

    Returns
    -------
    points : (N, 3) array
        Point positions.
    normals : (N, 3) array
        Point normals (pointing up in Z).
    """
    points = np.zeros((n_points, 3))
    points[:, 0] = np.random.uniform(-1, 1, n_points)
    points[:, 1] = np.random.uniform(-1, 1, n_points)
    points[:, 2] = np.random.randn(n_points) * noise

    normals = np.zeros((n_points, 3))
    normals[:, 2] = 1.0

    return points, normals


def load_xyz_file(filename):
    """
    Load point cloud from XYZ format (position xyz + normal xyz).

    Format: Each line contains: x y z nx ny nz

    Parameters
    ----------
    filename : str or Path
        Path to XYZ file.

    Returns
    -------
    points : (N, 3) array
        Point positions.
    normals : (N, 3) array
        Point normals.
    """
    filename = Path(filename)

    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split()]
                if len(values) >= 6:
                    data.append(values[:6])

    data = np.array(data)
    points = data[:, :3]
    normals = data[:, 3:6]

    # Normalize normals
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    # Avoid division by zero
    norm = np.where(norm > 0, norm, 1.0)
    normals /= norm

    return points, normals


def save_points_ply_ascii(points, normals, filename):
    """
    Save point cloud to ASCII PLY format.

    Parameters
    ----------
    points : (N, 3) array
        Point positions.
    normals : (N, 3) array
        Point normals.
    filename : str or Path
        Output PLY file path.
    """
    filename = Path(filename)
    n_points = len(points)

    header = f"""ply
format ascii 1.0
element vertex {n_points}
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


def save_points_ply_binary(points, normals, filename):
    """
    Save point cloud to binary PLY format (little-endian).

    This format is compatible with the PoissonRecon reference executable.

    Parameters
    ----------
    points : (N, 3) array
        Point positions.
    normals : (N, 3) array
        Point normals.
    filename : str or Path
        Output PLY file path.
    """
    filename = Path(filename)
    n_points = len(points)

    header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
end_header
""".encode('ascii')

    with open(filename, 'wb') as f:
        f.write(header)

        for p, n in zip(points, normals):
            # Write as little-endian floats
            f.write(struct.pack('<fff', p[0], p[1], p[2]))
            f.write(struct.pack('<fff', n[0], n[1], n[2]))


def convert_xyz_to_ply(xyz_file, ply_file, binary=True):
    """
    Convert XYZ format point cloud to PLY format.

    Parameters
    ----------
    xyz_file : str or Path
        Input XYZ file path.
    ply_file : str or Path
        Output PLY file path.
    binary : bool
        If True, write binary PLY; otherwise ASCII (default True).
    """
    points, normals = load_xyz_file(xyz_file)

    if binary:
        save_points_ply_binary(points, normals, ply_file)
    else:
        save_points_ply_ascii(points, normals, ply_file)


def load_mesh_ply(filename):
    """
    Load mesh from PLY file (vertices and faces).

    Parameters
    ----------
    filename : str or Path
        Path to PLY file.

    Returns
    -------
    vertices : (V, 3) array
        Vertex positions.
    faces : (F, 3) array
        Face indices.
    """
    filename = Path(filename)

    with open(filename, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse header
        n_vertices = 0
        n_faces = 0
        is_binary = False

        for line in header_lines:
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
            elif line.startswith('format binary'):
                is_binary = True

        # Read vertex data
        vertices = []
        for _ in range(n_vertices):
            if is_binary:
                data = f.read(12)  # 3 floats
                x, y, z = struct.unpack('<fff', data)
            else:
                line = f.readline().decode('ascii').strip()
                x, y, z = map(float, line.split())
            vertices.append([x, y, z])

        # Read face data
        faces = []
        for _ in range(n_faces):
            if is_binary:
                # Read vertex count (1 byte)
                n_verts = struct.unpack('<B', f.read(1))[0]
                # Read vertex indices (int32)
                data = f.read(n_verts * 4)
                indices = struct.unpack(f'<{n_verts}i', data)
            else:
                line = f.readline().decode('ascii').strip()
                parts = list(map(int, line.split()))
                n_verts = parts[0]
                indices = parts[1:]

            # Triangulate if needed (for quads, etc.)
            for i in range(1, len(indices) - 1):
                faces.append([indices[0], indices[i], indices[i + 1]])

    return np.array(vertices), np.array(faces)


def generate_half_sphere_points(n_points=1000, radius=1.0, noise=0.0):
    """
    Generate random points on a half-sphere surface (Z > 0).

    This creates an open surface suitable for testing boundary condition
    differences between Neumann, Dirichlet, and Free boundaries.

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    radius : float
        Sphere radius (default 1.0).
    noise : float
        Gaussian noise standard deviation (default 0.0).

    Returns
    -------
    points : (N, 3) array
        Point positions (upper hemisphere only).
    normals : (N, 3) array
        Point normals (pointing outward).
    """
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi/2, n_points)  # Only upper hemisphere

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.stack([x, y, z], axis=1)

    if noise > 0:
        points += np.random.randn(n_points, 3) * noise

    normals = points.copy()
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return points, normals


if __name__ == "__main__":
    # Test data generation
    import sys

    print("Generating test data...")

    # Generate sphere
    points, normals = generate_sphere_points(1000)
    print(f"Generated sphere: {len(points)} points")
    save_points_ply_binary(points, normals, "test_sphere.ply")
    print("Saved test_sphere.ply")

    # Generate cube
    points, normals = generate_cube_points(200)
    print(f"Generated cube: {len(points)} points")
    save_points_ply_binary(points, normals, "test_cube.ply")
    print("Saved test_cube.ply")

    # Generate half-sphere
    points, normals = generate_half_sphere_points(1000)
    print(f"Generated half-sphere: {len(points)} points")
    save_points_ply_binary(points, normals, "test_half_sphere.ply")
    print("Saved test_half_sphere.ply")

    # Test XYZ loading if file provided
    if len(sys.argv) > 1:
        xyz_file = sys.argv[1]
        try:
            points, normals = load_xyz_file(xyz_file)
            print(f"Loaded {xyz_file}: {len(points)} points")
            save_points_ply_binary(points, normals, "test_converted.ply")
            print("Saved test_converted.ply")
        except Exception as e:
            print(f"Error loading XYZ file: {e}")
