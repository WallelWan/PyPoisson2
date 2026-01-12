import sys
import ctypes
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Constants
INT32_MAX = 2**31 - 1

class ParallelType:
    """Parallelization strategy constants."""
    AUTO = -1   # Auto-detect (OpenMP > Async)
    OPEN_MP = 0 # Force OpenMP
    ASYNC = 1   # Force C++ std::async
    NONE = 2    # Serial (debug only)

class BoundaryType:
    """Boundary condition type constants."""
    FREE = 0      # Free boundary
    DIRICHLET = 1 # Dirichlet boundary (zero at boundary)
    NEUMANN = 2   # Neumann boundary (zero gradient at boundary)

class ReconstructionParams(ctypes.Structure):
    """C-ABI compatible struct for reconstruction parameters."""
    _fields_ = [
        ("depth", ctypes.c_int),
        ("full_depth", ctypes.c_int),
        ("cg_depth", ctypes.c_int),
        ("iters", ctypes.c_int),
        ("degree", ctypes.c_int),
        ("scale", ctypes.c_double),
        ("samples_per_node", ctypes.c_double),
        ("point_weight", ctypes.c_double),
        ("confidence", ctypes.c_bool),
        ("verbose", ctypes.c_bool),
        ("parallel_type", ctypes.c_int),
        ("grid_depth", ctypes.c_int),
        
        # NEW: Core solver parameters (P0)
        ("exact_interpolation", ctypes.c_bool),
        ("show_residual", ctypes.c_bool),
        ("low_depth_cutoff", ctypes.c_double),

        # NEW: Solver depth controls (P0)
        ("width", ctypes.c_double),
        ("cg_solver_accuracy", ctypes.c_double),
        ("base_depth", ctypes.c_int),
        ("solve_depth", ctypes.c_int),
        ("kernel_depth", ctypes.c_int),
        ("base_v_cycles", ctypes.c_int),

        # NEW: Mesh extraction parameters (P0)
        ("force_manifold", ctypes.c_bool),
        ("polygon_mesh", ctypes.c_bool),

        # NEW: Grid output parameters (P0)
        ("primal_grid", ctypes.c_bool),
        ("linear_fit", ctypes.c_bool),
        ("grid_coordinates", ctypes.c_bool),

        # NEW: Boundary conditions (P2)
        ("boundary_type", ctypes.c_int),
        ("dirichlet_erode", ctypes.c_bool),

        # NEW: Advanced output parameters (P3)
        ("output_density", ctypes.c_bool),
        ("output_gradients", ctypes.c_bool),
    ]

class PoissonLibrary:
    """Manages loading of standard (int32) and big (int64) libraries."""

    def __init__(self):
        self._lib_std = None
        self._lib_big = None

    def _load_library(self, lib_name: str) -> ctypes.CDLL:
        """Finds and loads the shared library."""
        # Search paths: local directory, build directory, site-packages lib
        possible_paths = [
            Path(__file__).parent / "lib" / lib_name,                 # Package internal lib
            Path(__file__).parent.parent / "lib" / lib_name,          # Development / cmake install
            Path(__file__).parent.parent / "build" / lib_name,        # CMake build dir
            Path(sys.prefix) / "lib" / lib_name,                      # System install
            Path(".").resolve() / lib_name,                           # Current dir
        ]
        
        # Handle platform-specific extensions
        if sys.platform.startswith('darwin'):
            lib_name = lib_name.replace('.so', '.dylib')
        elif sys.platform.startswith('win'):
            lib_name = lib_name.replace('.so', '.dll')

        # Try to find the file
        found_path = None
        for p in possible_paths:
            if p.exists():
                found_path = p
                break

        if not found_path:
             # Fallback: let ctypes find it in system paths (LD_LIBRARY_PATH)
             try:
                 return ctypes.CDLL(lib_name)
             except OSError:
                 pass
             
             raise FileNotFoundError(
                f"Could not find library '{lib_name}'. Checked paths:\n" + 
                "\n".join(str(p) for p in possible_paths)
             )

        return ctypes.CDLL(str(found_path))

    def _setup_signatures(self, lib: ctypes.CDLL, index_type: type):
        """Configures ctypes function signatures."""
        
        # run_poisson_reconstruction
        lib.run_poisson_reconstruction.restype = ctypes.c_void_p
        lib.run_poisson_reconstruction.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # pts
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # nmls
            ctypes.c_size_t,      # count
            ReconstructionParams  # params
        ]

        # Getters
        lib.get_vertex_count.restype = ctypes.c_size_t
        lib.get_vertex_count.argtypes = [ctypes.c_void_p]

        lib.get_face_count.restype = ctypes.c_size_t
        lib.get_face_count.argtypes = [ctypes.c_void_p]

        lib.get_grid_res.restype = ctypes.c_int
        lib.get_grid_res.argtypes = [ctypes.c_void_p]

        lib.get_iso_value.restype = ctypes.c_double
        lib.get_iso_value.argtypes = [ctypes.c_void_p]

        # Data retrieval
        lib.get_vertices_data.restype = None
        lib.get_vertices_data.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        ]

        lib.get_faces_data.restype = None
        lib.get_faces_data.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=index_type, ndim=1, flags='C_CONTIGUOUS')
        ]

        lib.get_grid_data.restype = None
        lib.get_grid_data.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        ]

        # NEW (P3): Density and gradient getters
        lib.get_density_count.restype = ctypes.c_size_t
        lib.get_density_count.argtypes = [ctypes.c_void_p]

        lib.get_gradient_count.restype = ctypes.c_size_t
        lib.get_gradient_count.argtypes = [ctypes.c_void_p]

        lib.get_densities_data.restype = None
        lib.get_densities_data.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        ]

        lib.get_gradients_data.restype = None
        lib.get_gradients_data.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        ]

        # Cleanup
        lib.free_mesh.restype = None
        lib.free_mesh.argtypes = [ctypes.c_void_p]

    def get_library(self, num_points: int, depth: int, force_big: Optional[bool] = None) -> Tuple[ctypes.CDLL, type]:
        """
        Selects the appropriate library based on data size or user override.
        Returns (library_instance, index_dtype).
        """
        if force_big is not None:
            use_big = force_big
        else:
            # Estimate node count
            max_nodes = self._estimate_max_nodes(num_points, depth)
            use_big = max_nodes > (INT32_MAX * 0.8)

        if use_big:
            if self._lib_big is None:
                lib = self._load_library("libpypoisson_big.so")
                self._setup_signatures(lib, np.int64)
                self._lib_big = lib
            return self._lib_big, np.int64
        else:
            if self._lib_std is None:
                lib = self._load_library("libpypoisson_std.so")
                self._setup_signatures(lib, np.int32)
                self._lib_std = lib
            return self._lib_std, np.int32

    @staticmethod
    def _estimate_max_nodes(num_points: int, depth: int) -> int:
        """Estimates max octree nodes to decide on index type."""
        # Heuristic: num_points * expansion * depth_factor
        # Depth factor assumes tree expands significantly beyond base depth 5
        base_nodes = num_points * 4
        depth_expansion = 2 ** max(0, depth - 5)
        return base_nodes * depth_expansion

# Singleton instance
_library_manager = PoissonLibrary()