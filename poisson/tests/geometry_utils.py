"""
Geometry utilities for mesh comparison and analysis.

Provides functions to compare reconstructed meshes, compute metrics,
and analyze mesh properties.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Compute the volume of a closed triangular mesh using signed tetrahedron method.

    Parameters
    ----------
    vertices : (V, 3) array
        Vertex positions.
    faces : (F, 3) array
        Face indices (triangles).

    Returns
    -------
    volume : float
        Mesh volume (absolute value).
    """
    # Get vertices for each face
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute signed volume of each tetrahedron (with origin as fourth point)
    # V = 1/6 * det([v0, v1, v2])
    cross = np.cross(v1, v2)
    dots = np.sum(v0 * cross, axis=1)
    signed_volume = np.sum(dots) / 6.0

    return abs(signed_volume)


def mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Compute the surface area of a triangular mesh.

    Parameters
    ----------
    vertices : (V, 3) array
        Vertex positions.
    faces : (F, 3) array
        Face indices (triangles).

    Returns
    -------
    area : float
        Total surface area.
    """
    # Get vertices for each face
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute triangle areas using cross product
    edges1 = v1 - v0
    edges2 = v2 - v0
    cross = np.cross(edges1, edges2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    return np.sum(areas)


def bounding_box(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the axis-aligned bounding box of a point set or mesh.

    Parameters
    ----------
    vertices : (V, 3) array
        Vertex positions.

    Returns
    -------
    min_bounds : (3,) array
        Minimum coordinates.
    max_bounds : (3,) array
        Maximum coordinates.
    """
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


def bounding_box_diagonal(vertices: np.ndarray) -> float:
    """
    Compute the diagonal length of the bounding box.

    Parameters
    ----------
    vertices : (V, 3) array
        Vertex positions.

    Returns
    -------
    diagonal : float
        Bounding box diagonal length.
    """
    min_b, max_b = bounding_box(vertices)
    return np.linalg.norm(max_b - min_b)


def hausdorff_distance(v1: np.ndarray, v2: np.ndarray, percentile: float = 100) -> float:
    """
    Compute the (approximate) Hausdorff distance between two point sets.

    Uses KD-tree for efficient nearest neighbor queries.

    Parameters
    ----------
    v1 : (N, 3) array
        First point set.
    v2 : (M, 3) array
        Second point set.
    percentile : float
        Use percentile instead of max for robustness (default 100 = true Hausdorff).
        Lower values (e.g., 95) ignore outliers.

    Returns
    -------
    distance : float
        Hausdorff distance (or percentile distance).
    """
    # Build KD-tree for v2
    tree2 = cKDTree(v2)

    # Compute distances from v1 to v2
    distances1 = tree2.query(v1)[0]

    # Build KD-tree for v1
    tree1 = cKDTree(v1)

    # Compute distances from v2 to v1
    distances2 = tree1.query(v2)[0]

    # Hausdorff is the max of both directions
    h1 = np.percentile(distances1, percentile)
    h2 = np.percentile(distances2, percentile)

    return max(h1, h2)


def chamfer_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the Chamfer distance between two point sets.

    Chamfer distance is the average of squared distances in both directions.

    Parameters
    ----------
    v1 : (N, 3) array
        First point set.
    v2 : (M, 3) array
        Second point set.

    Returns
    -------
    distance : float
        Chamfer distance.
    """
    # Build KD-trees
    tree1 = cKDTree(v1)
    tree2 = cKDTree(v2)

    # Compute nearest neighbor distances
    distances1 = tree2.query(v1)[0]
    distances2 = tree1.query(v2)[0]

    # Chamfer distance
    return np.mean(distances1**2) + np.mean(distances2**2)


def compare_meshes(v1: np.ndarray, f1: np.ndarray,
                   v2: np.ndarray, f2: np.ndarray,
                   tolerance: float = 1e-6) -> dict:
    """
    Compare two meshes and return various metrics.

    Parameters
    ----------
    v1, f1 : (V1, 3), (F1, 3) arrays
        First mesh.
    v2, f2 : (V2, 3), (F2, 3) arrays
        Second mesh.
    tolerance : float
        Numerical tolerance for vertex comparison (default 1e-6).

    Returns
    -------
    metrics : dict
        Dictionary containing comparison metrics:
        - 'vertex_count_diff': Difference in vertex count
        - 'face_count_diff': Difference in face count
        - 'volume_diff': Relative volume difference
        - 'surface_area_diff': Relative surface area difference
        - 'hausdorff': Hausdorff distance
        - 'chamfer': Chamfer distance
        - 'bbox_diagonal_diff': Bounding box diagonal difference
    """
    metrics = {}

    # Count differences
    metrics['vertex_count_diff'] = len(v2) - len(v1)
    metrics['face_count_diff'] = len(f2) - len(f1)

    # Volume comparison
    vol1 = mesh_volume(v1, f1)
    vol2 = mesh_volume(v2, f2)
    if vol1 > 0:
        metrics['volume_diff'] = (vol2 - vol1) / vol1
    else:
        metrics['volume_diff'] = 0.0

    # Surface area comparison
    area1 = mesh_surface_area(v1, f1)
    area2 = mesh_surface_area(v2, f2)
    if area1 > 0:
        metrics['surface_area_diff'] = (area2 - area1) / area1
    else:
        metrics['surface_area_diff'] = 0.0

    # Point cloud distances
    metrics['hausdorff'] = hausdorff_distance(v1, v2, percentile=95)
    metrics['chamfer'] = chamfer_distance(v1, v2)

    # Bounding box comparison
    bbox1 = bounding_box_diagonal(v1)
    bbox2 = bounding_box_diagonal(v2)
    if bbox1 > 0:
        metrics['bbox_diagonal_diff'] = (bbox2 - bbox1) / bbox1
    else:
        metrics['bbox_diagonal_diff'] = 0.0

    return metrics


def check_manifold(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """
    Check if a mesh is manifold (watertight).

    Parameters
    ----------
    vertices : (V, 3) array
        Vertex positions.
    faces : (F, 3) array
        Face indices.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'is_manifold': Boolean indicating if mesh is manifold
        - 'non_manifold_edges': Count of non-manifold edges
        - 'boundary_edges': Count of boundary edges
    """
    # Count edge occurrences
    edge_count = {}
    for face in faces:
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Manifold mesh: each edge appears exactly twice
    non_manifold_edges = sum(1 for c in edge_count.values() if c != 2)
    boundary_edges = sum(1 for c in edge_count.values() if c == 1)

    return {
        'is_manifold': non_manifold_edges == 0,
        'non_manifold_edges': non_manifold_edges,
        'boundary_edges': boundary_edges
    }


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute face normals for a mesh.

    Parameters
    ----------
    vertices : (V, 3) array
        Vertex positions.
    faces : (F, 3) array
        Face indices.

    Returns
    -------
    normals : (F, 3) array
        Face normals (normalized).
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edges1 = v1 - v0
    edges2 = v2 - v0
    cross = np.cross(edges1, edges2)

    # Normalize
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normals = cross / norms

    return normals


def spherical_symmetry_error(vertices: np.ndarray, center: Optional[np.ndarray] = None) -> float:
    """
    Compute spherical symmetry error for a point set.

    Measures how much the points deviate from a perfect sphere centered at `center`.

    Parameters
    ----------
    vertices : (V, 3) array
        Vertex positions.
    center : (3,) array, optional
        Sphere center. If None, uses centroid.

    Returns
    -------
    error : float
        Relative standard deviation of radial distances.
    """
    if center is None:
        center = np.mean(vertices, axis=0)

    # Compute radii
    radii = np.linalg.norm(vertices - center, axis=1)
    mean_radius = np.mean(radii)

    if mean_radius > 0:
        return np.std(radii) / mean_radius
    return 0.0


def parse_iso_value_from_output(output_str: str) -> Optional[float]:
    """
    Parse iso-value from PoissonRecon executable output.

    Looks for pattern like "Iso-Value: 0.500123"

    Parameters
    ----------
    output_str : str
        Stdout/stderr from PoissonRecon executable.

    Returns
    -------
    iso_value : float or None
        Extracted iso-value, or None if not found.
    """
    import re

    match = re.search(r"Iso-Value:\s+([0-9eE.+-]+)", output_str)
    if match:
        return float(match.group(1))
    return None


def format_metrics_report(metrics: dict, title: str = "Mesh Comparison") -> str:
    """
    Format metrics dictionary as a readable report.

    Parameters
    ----------
    metrics : dict
        Comparison metrics from compare_meshes().
    title : str
        Report title.

    Returns
    -------
    report : str
        Formatted report string.
    """
    lines = [f"{title}", "=" * len(title)]

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.6e}")
        else:
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test geometry utilities
    print("Testing geometry utilities...")

    # Generate test sphere
    from test_data_generator import generate_sphere_points
    points, normals = generate_sphere_points(1000)

    # Create simple triangulation (not a proper mesh, just for testing)
    n = len(points)
    faces = np.array([
        [i, (i + 1) % n, (i + 2) % n]
        for i in range(0, n - 2, 3)
    ])

    print(f"Mesh: {len(points)} vertices, {len(faces)} faces")
    print(f"Volume: {mesh_volume(points, faces):.6f}")
    print(f"Surface area: {mesh_surface_area(points, faces):.6f}")
    print(f"Bounding box diagonal: {bounding_box_diagonal(points):.6f}")
    print(f"Spherical symmetry error: {spherical_symmetry_error(points):.6f}")

    manifold = check_manifold(points, faces)
    print(f"Is manifold: {manifold['is_manifold']}")
