import numpy as np
from .core import _library_manager, ReconstructionParams, ParallelType, BoundaryType

def poisson_reconstruction(
    points: np.ndarray,
    normals: np.ndarray,
    depth: int = 8,
    full_depth: int = 5,
    scale: float = 1.1,
    samples_per_node: float = 1.5,
    cg_depth: int = 0,
    iters: int = 8,
    degree: int = 1,
    point_weight: float = 4.0,
    confidence: bool = False,
    verbose: bool = False,
    parallel_type: int = ParallelType.AUTO,
    grid_depth: int = 0,

    # NEW: Core solver parameters (P0)
    exact_interpolation: bool = False,
    show_residual: bool = False,
    low_depth_cutoff: float = 0.0,

    # NEW: Solver depth controls (P0)
    width: float = 0.0,
    cg_solver_accuracy: float = 1e-3,
    base_depth: int = -1,
    solve_depth: int = -1,
    kernel_depth: int = -1,
    base_v_cycles: int = 1,

    # NEW: Mesh extraction parameters (P0)
    force_manifold: bool = True,
    polygon_mesh: bool = False,

    # NEW: Grid output parameters (P0)
    primal_grid: bool = False,
    linear_fit: bool = False,
    grid_coordinates: bool = False,

    # NEW: Boundary conditions (P2)
    boundary: str = 'neumann', # 'neumann', 'dirichlet', 'free'
    dirichlet_erode: bool = False,

    # NEW: Advanced output parameters (P3)
    output_density: bool = False,
    output_gradients: bool = False,

    # NEW: Runtime safety / indexing controls
    validate_finite: bool = True,
    force_big: bool | None = None,
) -> tuple:
    """
    Perform Screened Poisson Surface Reconstruction.

    Automatically selects between 32-bit and 64-bit indexing based on problem size.

    Parameters
    ----------
    points : (N, 3) float64 array
        Input point cloud positions.
    normals : (N, 3) float64 array
        Input point cloud normals (normalized).
    depth : int, optional
        Maximum tree depth (default 8). Determines resolution.
    full_depth : int, optional
        Depth of the complete octree (default 5).
    scale : float, optional
        Bounding box scale factor (default 1.1).
    samples_per_node : float, optional
        Minimum samples per octree node (default 1.5).
    cg_depth : int, optional
        Depth for Conjugate Gradients solver (default 0).
    iters : int, optional
        Solver iterations (default 8).
    degree : int, optional
        B-Spline degree, usually 1 or 2 (default 1).
    point_weight : float, optional
        Weight of point interpolation constraints (default 4.0).
    confidence : bool, optional
        Use normal magnitude as confidence weight (default False).
    verbose : bool, optional
        Print debug info to stdout (default False).
    parallel_type : int, optional
        Parallelization strategy (default ParallelType.AUTO).
        0=OpenMP, 1=Async, 2=Serial.
    grid_depth : int, optional
        If > 0, also returns the implicit field evaluated on a regular grid 
         of resolution 2^grid_depth.
    exact_interpolation : bool, optional
        Prefer exact interpolation (default False).
    show_residual : bool, optional
        Show solver residual (default False).
    low_depth_cutoff : float, optional
        Low depth cutoff (default 0.0).
    width : float, optional
        Target width (overrides depth if > 0) (default 0.0).
    cg_solver_accuracy : float, optional
        Conjugate Gradients solver accuracy (default 1e-3).
    base_depth : int, optional
        Base depth (default -1).
    solve_depth : int, optional
        Solve depth (default -1).
    kernel_depth : int, optional
        Kernel depth (default -1).
    base_v_cycles : int, optional
        Base V-Cycles (default 1).
    force_manifold : bool, optional
        Force manifold output (default True).
    polygon_mesh : bool, optional
        Output polygon mesh (default False).
    primal_grid : bool, optional
        Use primal grid (default False).
    linear_fit : bool, optional
        Use linear fit (default False).
    grid_coordinates : bool, optional
        Use grid coordinates (default False).
    boundary : str, optional
        Boundary condition type: 'neumann' (default), 'dirichlet', or 'free'.
    dirichlet_erode : bool, optional
        Use Dirichlet erosion (default False).
    output_density : bool, optional
        Output per-vertex density values (default False).
        If True, returns an additional (V,) array of density values.
    output_gradients : bool, optional
        Output per-vertex gradient vectors (default False).
        If True, returns an additional (V, 3) array of gradient vectors.
    validate_finite : bool, optional
        Validate points/normals for NaN/Inf values (default True).
    force_big : bool, optional
        Force 64-bit indexing library (default None = auto).

    Returns
    -------
    vertices : (V, 3) float64 array
        Reconstructed mesh vertices.
    faces : (F, 3) int32 or int64 array
        Reconstructed mesh faces.
    grid : (R, R, R) float64 array, optional
        Implicit field values on a regular grid. Only returned if grid_depth > 0.
    iso_value : float, optional
        The iso-value used for mesh extraction (typically ~0.5). Only returned if grid_depth > 0.
    densities : (V,) float64 array, optional
        Per-vertex density values. Only returned if output_density=True.
    gradients : (V, 3) float64 array, optional
        Per-vertex gradient vectors. Only returned if output_gradients=True.
    """
    # 1. Input Validation
    points = np.ascontiguousarray(points, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3) float64 array")
    if normals.shape != points.shape:
        raise ValueError("normals must have same shape as points")
    if validate_finite:
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite (no NaN/Inf)")
        if not np.all(np.isfinite(normals)):
            raise ValueError("normals must be finite (no NaN/Inf)")

    num_points = points.shape[0]
    if num_points == 0:
        res = [np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)]
        if grid_depth > 0:
            res.extend([np.zeros((0, 0, 0), dtype=np.float64), 0.0])
        if output_density:
            res.append(np.zeros((0,), dtype=np.float64))
        if output_gradients:
            res.append(np.zeros((0, 3), dtype=np.float64))
        return tuple(res)

    # 2. Select Library (Auto int32/int64)
    lib, face_dtype = _library_manager.get_library(num_points, depth, force_big=force_big)

    if verbose:
        print(f"[Pypoisson] Processing {num_points} points at depth {depth}.")
        print(f"[Pypoisson] Using {'int64' if face_dtype == np.int64 else 'int32'} indices.")

    # 3. Prepare Parameters
    params = ReconstructionParams()
    params.depth = depth
    params.full_depth = full_depth
    params.cg_depth = cg_depth
    params.iters = iters
    params.degree = degree
    params.scale = scale
    params.samples_per_node = samples_per_node
    params.point_weight = point_weight
    params.confidence = confidence
    params.verbose = verbose
    params.parallel_type = parallel_type
    params.grid_depth = grid_depth
    
    params.exact_interpolation = exact_interpolation
    params.show_residual = show_residual
    params.low_depth_cutoff = low_depth_cutoff
    params.width = width
    params.cg_solver_accuracy = cg_solver_accuracy
    params.base_depth = base_depth
    params.solve_depth = solve_depth
    params.kernel_depth = kernel_depth
    params.base_v_cycles = base_v_cycles
    params.force_manifold = force_manifold
    params.polygon_mesh = polygon_mesh
    params.primal_grid = primal_grid
    params.linear_fit = linear_fit
    params.grid_coordinates = grid_coordinates

    # Map boundary string to int
    b_lower = boundary.lower()
    if b_lower == 'neumann':
        params.boundary_type = BoundaryType.NEUMANN
    elif b_lower == 'dirichlet':
        params.boundary_type = BoundaryType.DIRICHLET
    elif b_lower == 'free':
        params.boundary_type = BoundaryType.FREE
    else:
        raise ValueError(f"Unknown boundary type: '{boundary}'. Expected 'neumann', 'dirichlet', or 'free'.")

    params.dirichlet_erode = dirichlet_erode

    # NEW (P3): Advanced output parameters
    params.output_density = output_density
    params.output_gradients = output_gradients

    # 4. Call C++
    # Keep references to arrays to prevent GC during C++ call (though ctypes usually handles this in argtypes)
    mesh_handle = lib.run_poisson_reconstruction(points, normals, num_points, params)

    if not mesh_handle:
        raise RuntimeError("Poisson reconstruction returned NULL handle.")

    try:
        # 5. Retrieve Mesh Data
        v_count = lib.get_vertex_count(mesh_handle)
        f_count = lib.get_face_count(mesh_handle)

        vertices = np.zeros(v_count * 3, dtype=np.float64)
        faces = np.zeros(f_count * 3, dtype=face_dtype)

        if v_count > 0:
            lib.get_vertices_data(mesh_handle, vertices)
        if f_count > 0:
            lib.get_faces_data(mesh_handle, faces)

        # Reshape to (N, 3)
        vertices = vertices.reshape((-1, 3))
        faces = faces.reshape((-1, 3))

        # 6. Retrieve Grid Data if requested
        result = [vertices, faces]

        if grid_depth > 0:
            res = lib.get_grid_res(mesh_handle)
            if res > 0:
                grid = np.zeros(res * res * res, dtype=np.float64)
                lib.get_grid_data(mesh_handle, grid)
                grid = grid.reshape((res, res, res))

                iso_value = lib.get_iso_value(mesh_handle)
                result.extend([grid, iso_value])
            else:
                result.extend([np.zeros((0, 0, 0), dtype=np.float64), 0.0])

        # 7. Retrieve Density and Gradient Data if requested (P3)
        if output_density:
            d_count = lib.get_density_count(mesh_handle)
            if d_count > 0:
                densities = np.zeros(d_count, dtype=np.float64)
                lib.get_densities_data(mesh_handle, densities)
                result.append(densities)
            else:
                result.append(np.zeros((0,), dtype=np.float64))

        if output_gradients:
            g_count = lib.get_gradient_count(mesh_handle)
            if g_count > 0:
                gradients = np.zeros(g_count * 3, dtype=np.float64)
                lib.get_gradients_data(mesh_handle, gradients)
                gradients = gradients.reshape((-1, 3))
                result.append(gradients)
            else:
                result.append(np.zeros((0, 3), dtype=np.float64))

        return tuple(result)

    finally:
        # 8. Cleanup
        lib.free_mesh(mesh_handle)
