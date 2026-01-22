#ifndef PYPOISSON_WRAPPER_API_H
#define PYPOISSON_WRAPPER_API_H

#include <cstddef>

struct ReconstructionParams {
    int depth;
    int full_depth;
    int cg_depth;
    int iters;
    int degree;
    double scale;
    double samples_per_node;
    double point_weight;
    bool confidence;
    bool verbose;
    int parallel_type;
    int grid_depth;

    bool exact_interpolation;
    bool show_residual;
    double low_depth_cutoff;

    double width;
    double cg_solver_accuracy;
    int base_depth;
    int solve_depth;
    int kernel_depth;
    int base_v_cycles;

    bool force_manifold;
    bool polygon_mesh;

    bool primal_grid;
    bool linear_fit;
    bool grid_coordinates;

    int boundary_type;
    bool dirichlet_erode;

    bool output_density;
    bool output_gradients;
};

extern "C" {
    void* run_poisson_reconstruction(double* pts, double* nmls, size_t count, ReconstructionParams params);
    size_t get_vertex_count(void* mesh);
    size_t get_face_count(void* mesh);
    int get_grid_res(void* mesh);
    double get_iso_value(void* mesh);
    void get_vertices_data(void* mesh, double* buffer);
    void get_faces_data(void* mesh, void* buffer);
    void get_grid_data(void* mesh, double* buffer);
    size_t get_density_count(void* mesh);
    size_t get_gradient_count(void* mesh);
    void get_densities_data(void* mesh, double* buffer);
    void get_gradients_data(void* mesh, double* buffer);
    void free_mesh(void* mesh);
}

#endif
