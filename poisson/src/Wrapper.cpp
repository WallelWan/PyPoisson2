#include <vector>
#include <cstring>
#include <iostream>

// Export macro for symbol visibility
#if defined(_WIN32)
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

// Define FAST_COMPILE to speed up compilation by restricting B-Spline degrees to 1 and 2
#ifndef FAST_COMPILE
#define FAST_COMPILE
#endif

// Include PoissonRecon headers
#include "PreProcessor.h"
#include "Reconstructors.h"
#include "Geometry.h"
#include "MyMiscellany.h"
#include "MultiThreading.h"
#include "WrapperApi.h"

using namespace PoissonRecon;

// =================================================================================
// 1. Adapter Classes (Bridging Numpy <-> C++ Library)
// =================================================================================

// Input Stream: Reads points/normals from flat Numpy arrays
template< typename Real , unsigned int Dim >
struct NumpyInputSampleStream : public Reconstructor::InputOrientedSampleStream< Real , Dim >
{
    const Real* _points;
    const Real* _normals;
    size_t _current;
    size_t _count;

    NumpyInputSampleStream(const Real* points, const Real* normals, size_t count)
        : _points(points), _normals(normals), _count(count), _current(0) {}

    // CRITICAL: The library reads data twice (once for bounds, once for tree building)
    void reset( void ) { _current = 0; }

    bool read( Point< Real , Dim > &p , Point< Real , Dim > &n )
    {
        if( _current < _count )
        {
            for( unsigned int d=0 ; d<Dim ; d++ )
            {
                p[d] = _points[ _current*Dim + d ];
                n[d] = _normals[ _current*Dim + d ];
            }
            _current++;
            return true;
        }
        else return false;
    }
};

// Output Vertex Stream: Pushes vertices to a std::vector
template< typename Real , unsigned int Dim >
struct NumpyOutputVertexStream : public Reconstructor::OutputLevelSetVertexStream< Real , Dim >
{
    std::vector< Real > vertices_flat; // Flattened storage [x,y,z, x,y,z, ...]
    std::vector< Real > densities;     // NEW (P3): Per-vertex density values
    std::vector< Real > gradients_flat; // NEW (P3): Per-vertex gradients (3*N vertices)

    // Store reference to parent for callback
    ReconstructionParams* params;

    NumpyOutputVertexStream(ReconstructionParams* p) : params(p) {}

    // Override the pure virtual method
    size_t size( void ) const { return vertices_flat.size() / Dim; }

    // CRITICAL SIGNATURE: Must match library exactly
    size_t write( const Point< Real , Dim > &p , const Point< Real , Dim > &g , const Real &w )
    {
        for( unsigned int d=0 ; d<Dim ; d++ ) vertices_flat.push_back( p[d] );

        // NEW (P3): Capture density and gradients if requested
        if(params->output_density) {
            densities.push_back( w );
        }
        if(params->output_gradients) {
            for( unsigned int d=0 ; d<Dim ; d++ ) {
                gradients_flat.push_back( g[d] );
            }
        }

        return (vertices_flat.size() / Dim) - 1;
    }
};

// Output Face Stream: Pushes triangle indices to a std::vector
struct NumpyOutputFaceStream : public Reconstructor::OutputFaceStream< 2 > // 2 = dimension of the simplex (triangle)
{
    // Flattened storage [v1,v2,v3, v1,v2,v3, ...]
    // Note: node_index_type depends on BIG_DATA macro (int or long long)
    std::vector< node_index_type > faces_flat; 

    size_t size( void ) const { return faces_flat.size() / 3; }

    size_t write( const std::vector< node_index_type > &polygon )
    {
        if( polygon.size() != 3 ) 
        {
            // Should not happen for marching cubes, but good to be safe
            return -1; 
        }
        faces_flat.push_back( polygon[0] );
        faces_flat.push_back( polygon[1] );
        faces_flat.push_back( polygon[2] );
        return (faces_flat.size() / 3) - 1;
    }
};

// =================================================================================
// 2. Data Structures for C-ABI
// =================================================================================

// Opaque handle to store the result in C++ heap
struct MeshHandle {
    std::vector<double> vertices;      // Flattened vertices
    std::vector<node_index_type> faces; // Flattened faces (int or long long)
    std::vector<double> grid_values;   // Voxel grid values
    int grid_res;                      // Voxel grid resolution
    double iso_value;                  // Iso-value used for extraction

    // NEW: Optional additional outputs (P3)
    std::vector<double> densities;     // Per-vertex density values
    std::vector<double> gradients;     // Per-vertex gradients (3*N vertices)
};

// =================================================================================
// 3. Template Dispatch Logic
// =================================================================================

// Helper to run reconstruction for a specific B-Spline degree and Boundary Type
template< unsigned int Degree , BoundaryType BType >
MeshHandle* execute_reconstruction(double* pts, double* nmls, size_t count, ReconstructionParams params)
{
    // 1. Setup Input Stream
    NumpyInputSampleStream<double, 3> pointStream(pts, nmls, count);

    // 2. Configure Solver Parameters
    typename Reconstructor::Poisson::SolutionParameters<double> solverParams;
    solverParams.verbose = params.verbose;
    solverParams.depth = (unsigned int)params.depth;
    solverParams.fullDepth = (unsigned int)params.full_depth;
    solverParams.iters = (unsigned int)params.iters;
    solverParams.scale = (double)params.scale;
    solverParams.samplesPerNode = (double)params.samples_per_node;
    solverParams.pointWeight = (double)params.point_weight;
    solverParams.confidence = params.confidence;

    // NEW: Map P0 parameters
    solverParams.exactInterpolation = params.exact_interpolation;
    solverParams.showResidual = params.show_residual;
    solverParams.lowDepthCutOff = params.low_depth_cutoff;
    solverParams.width = params.width;
    solverParams.cgSolverAccuracy = params.cg_solver_accuracy;
    solverParams.baseDepth = (unsigned int)params.base_depth;
    solverParams.solveDepth = (unsigned int)params.solve_depth;
    solverParams.kernelDepth = (unsigned int)params.kernel_depth;
    solverParams.baseVCycles = (unsigned int)params.base_v_cycles;

    // NEW: Map P2 parameters
    solverParams.dirichletErode = params.dirichlet_erode;
    
    // 3. Type Definitions
    static const unsigned int Dim = 3;
    // Define FEM Signature based on Degree and Boundary Type
    static const unsigned int FEMSig = FEMDegreeAndBType< Degree , BType >::Signature;
    using FEMSigs = IsotropicUIntPack< Dim , FEMSig >;
    
    // 4. Run Solver
    using Solver = typename Reconstructor::Poisson::template Solver< double , Dim , FEMSigs >;
    using Implicit = typename Reconstructor::template Implicit< double , Dim , FEMSigs >;

    Implicit* implicit = Solver::Solve( pointStream , solverParams );

    // 5. Extract Mesh
    Reconstructor::LevelSetExtractionParameters extractionParams;
    extractionParams.linearFit = params.linear_fit;
    extractionParams.outputGradients = params.output_gradients;  // NEW (P3): Enable gradient output
    extractionParams.gridCoordinates = params.grid_coordinates;
    extractionParams.verbose = params.verbose;
    extractionParams.forceManifold = params.force_manifold;
    extractionParams.polygonMesh = params.polygon_mesh;

    // Output Streams - pass params pointer (NEW for P3)
    NumpyOutputVertexStream<double, Dim> vStream(&params);
    NumpyOutputFaceStream fStream;

    implicit->extractLevelSet( vStream , fStream , extractionParams );

    // 6. Pack Results into Handle
    MeshHandle* handle = new MeshHandle();
    handle->vertices.swap(vStream.vertices_flat);
    handle->faces.swap(fStream.faces_flat);
    handle->iso_value = implicit->isoValue;

    // NEW (P3): Pack densities and gradients if requested
    handle->densities.swap(vStream.densities);
    handle->gradients.swap(vStream.gradients_flat);

    // 7. Extract Implicit Field (Voxel Grid) if requested
    handle->grid_res = 0;
    if (params.grid_depth > 0) {
        int res = 0;
        // Evaluate on a regular grid. primal=false (centered voxels)
        Pointer(double) values = implicit->tree.template regularGridEvaluate< true >( implicit->solution , res , params.grid_depth , params.primal_grid );
        if (values) {
            size_t total_voxels = (size_t)res * res * res;
            handle->grid_values.resize(total_voxels);
            std::memcpy(handle->grid_values.data(), values, total_voxels * sizeof(double));
            handle->grid_res = res;
            DeletePointer(values);
        }
    }

    // Cleanup
    delete implicit;
    return handle;
}

// =================================================================================
// 4. Exported C Functions
// =================================================================================

// Helper to dispatch based on Boundary Type for a fixed Degree
template< unsigned int Degree >
MeshHandle* dispatch_boundary(double* pts, double* nmls, size_t count, ReconstructionParams params) {
    switch(params.boundary_type) {
        case 0: return execute_reconstruction<Degree, BOUNDARY_FREE>(pts, nmls, count, params);
        case 1: return execute_reconstruction<Degree, BOUNDARY_DIRICHLET>(pts, nmls, count, params);
        case 2: return execute_reconstruction<Degree, BOUNDARY_NEUMANN>(pts, nmls, count, params);
        default:
            if(params.verbose) std::cerr << "[Error] Unknown boundary type: " << params.boundary_type << std::endl;
            return nullptr;
    }
}

extern "C" {

// Helper to setup thread pool based on parallel_type
static void _setup_parallelization(int parallel_type) {
    if (parallel_type >= 0) {
        ThreadPool::ParallelizationType = (ThreadPool::ParallelType)parallel_type;
    } else {
        #ifdef _OPENMP
            ThreadPool::ParallelizationType = (ThreadPool::ParallelType)0; // OPEN_MP
        #else
            ThreadPool::ParallelizationType = (ThreadPool::ParallelType)1; // ASYNC
        #endif
    }
    // Default thread count = hardware concurrency
    // ThreadPool::NumThreads( std::thread::hardware_concurrency() ); // Library does this by default usually
}

EXPORT void* run_poisson_reconstruction(double* pts, double* nmls, size_t count, ReconstructionParams params)
{
    // Initialize Thread Pool
    _setup_parallelization(params.parallel_type);

    // Dispatch based on Degree
    // Note: PoissonRecon typically supports degrees 1, 2, 3, 4. 
    // FAST_COMPILE macro usually restricts this to 1 and 2.
    if (params.degree == 1) {
        return dispatch_boundary<1>(pts, nmls, count, params);
    } else if (params.degree == 2) {
        return dispatch_boundary<2>(pts, nmls, count, params);
    } 
    #ifndef FAST_COMPILE
    else if (params.degree == 3) {
        return dispatch_boundary<3>(pts, nmls, count, params);
    } else if (params.degree == 4) {
        return dispatch_boundary<4>(pts, nmls, count, params);
    }
    #endif
    else {
        if(params.verbose) std::cerr << "[Error] Unsupported B-Spline degree: " << params.degree << std::endl;
        return nullptr;
    }
}

EXPORT size_t get_vertex_count(void* mesh) {
    if(!mesh) return 0;
    return static_cast<MeshHandle*>(mesh)->vertices.size() / 3;
}

EXPORT size_t get_face_count(void* mesh) {
    if(!mesh) return 0;
    return static_cast<MeshHandle*>(mesh)->faces.size() / 3;
}

EXPORT int get_grid_res(void* mesh) {
    if(!mesh) return 0;
    return static_cast<MeshHandle*>(mesh)->grid_res;
}

EXPORT double get_iso_value(void* mesh) {
    if(!mesh) return 0.0;
    return static_cast<MeshHandle*>(mesh)->iso_value;
}

EXPORT void get_vertices_data(void* mesh, double* buffer) {
    if(!mesh || !buffer) return;
    MeshHandle* h = static_cast<MeshHandle*>(mesh);
    std::memcpy(buffer, h->vertices.data(), h->vertices.size() * sizeof(double));
}

EXPORT void get_faces_data(void* mesh, void* buffer) {
    if(!mesh || !buffer) return;
    MeshHandle* h = static_cast<MeshHandle*>(mesh);
    // Copy raw bytes. The caller must ensure 'buffer' has the correct type (int32 or int64)
    // depending on what 'node_index_type' is compiled as.
    std::memcpy(buffer, h->faces.data(), h->faces.size() * sizeof(node_index_type));
}

EXPORT void get_grid_data(void* mesh, double* buffer) {
    if(!mesh || !buffer) return;
    MeshHandle* h = static_cast<MeshHandle*>(mesh);
    std::memcpy(buffer, h->grid_values.data(), h->grid_values.size() * sizeof(double));
}

// NEW (P3): Density and gradient getter functions
EXPORT size_t get_density_count(void* mesh) {
    if(!mesh) return 0;
    return static_cast<MeshHandle*>(mesh)->densities.size();
}

EXPORT size_t get_gradient_count(void* mesh) {
    if(!mesh) return 0;
    return static_cast<MeshHandle*>(mesh)->gradients.size() / 3;
}

EXPORT void get_densities_data(void* mesh, double* buffer) {
    if(!mesh || !buffer) return;
    MeshHandle* h = static_cast<MeshHandle*>(mesh);
    std::memcpy(buffer, h->densities.data(), h->densities.size() * sizeof(double));
}

EXPORT void get_gradients_data(void* mesh, double* buffer) {
    if(!mesh || !buffer) return;
    MeshHandle* h = static_cast<MeshHandle*>(mesh);
    std::memcpy(buffer, h->gradients.data(), h->gradients.size() * sizeof(double));
}

EXPORT void free_mesh(void* mesh) {
    if(mesh) {
        delete static_cast<MeshHandle*>(mesh);
    }
}

} // extern "C"
