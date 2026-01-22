#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "PreProcessor.h"
#include "FEMTree.h"
#include "WrapperApi.h"

using PoissonRecon::node_index_type;

namespace {

struct DumpOptions {
    std::string input_ply;
    std::string output_ply;
    std::string density_out;
    std::string gradients_out;
    ReconstructionParams params{};
};

std::string DeriveOutputPath(const std::string& base, const std::string& suffix) {
    return base + suffix;
}

void PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --in <input.ply> --out <output.ply> [options]\n"
              << "Options:\n"
              << "  --density                Output density values\n"
              << "  --gradients              Output gradient vectors\n"
              << "  --densityOut <path>       Density output path (default: <out>.density.bin)\n"
              << "  --gradientsOut <path>     Gradients output path (default: <out>.gradients.bin)\n"
              << "  --depth <int>             Max depth (default 8)\n"
              << "  --fullDepth <int>         Full depth (default 5)\n"
              << "  --iters <int>             GS iterations (default 8)\n"
              << "  --degree <int>            B-spline degree (default 1)\n"
              << "  --scale <float>           Scale factor (default 1.1)\n"
              << "  --samplesPerNode <float>  Samples per node (default 1.5)\n"
              << "  --pointWeight <float>     Point weight (default 4.0)\n"
              << "  --confidence              Enable confidence weighting\n"
              << "  --verbose                 Verbose output\n"
              << "  --exact                   Exact interpolation\n"
              << "  --showResidual            Show residual\n"
              << "  --lowDepthCutOff <float>  Low depth cutoff\n"
              << "  --width <float>           Width override\n"
              << "  --cgAccuracy <float>      CG solver accuracy\n"
              << "  --baseDepth <int>         Base depth\n"
              << "  --solveDepth <int>        Solve depth\n"
              << "  --kernelDepth <int>       Kernel depth\n"
              << "  --baseVCycles <int>       Base V cycles\n"
              << "  --nonManifold             Allow non-manifold output\n"
              << "  --polygonMesh             Output polygon mesh\n"
              << "  --primalGrid              Use primal grid\n"
              << "  --linearFit               Use linear fit\n"
              << "  --gridCoordinates         Output grid coordinates\n"
              << "  --bType <1|2|3>            Boundary type (1=free,2=dirichlet,3=neumann)\n"
              << "  --dirichletErode          Enable dirichlet erosion\n"
              << "  --noErode                 Disable dirichlet erosion\n"
              << "  --parallel <int>          Parallel type (-1 auto,0 OpenMP,1 async,2 none)\n";
}

bool ReadAsciiPly(const std::string& path, std::vector<double>& points, std::vector<double>& normals) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open input: " << path << "\n";
        return false;
    }

    std::string line;
    bool format_ascii = false;
    size_t vertex_count = 0;
    bool in_header = true;

    while (in_header && std::getline(in, line)) {
        if (line.rfind("format ", 0) == 0) {
            format_ascii = (line.find("ascii") != std::string::npos);
        } else if (line.rfind("element vertex", 0) == 0) {
            std::istringstream iss(line);
            std::string elem, vertex;
            iss >> elem >> vertex >> vertex_count;
        } else if (line == "end_header") {
            in_header = false;
        }
    }

    if (!format_ascii) {
        std::cerr << "Only ASCII PLY is supported by PoissonReconDump.\n";
        return false;
    }
    if (vertex_count == 0) {
        std::cerr << "No vertices found in PLY header.\n";
        return false;
    }

    points.resize(vertex_count * 3);
    normals.resize(vertex_count * 3);

    for (size_t i = 0; i < vertex_count; ++i) {
        if (!std::getline(in, line)) {
            std::cerr << "Unexpected EOF while reading vertices.\n";
            return false;
        }
        std::istringstream iss(line);
        double x, y, z, nx, ny, nz;
        if (!(iss >> x >> y >> z >> nx >> ny >> nz)) {
            std::cerr << "Failed to parse vertex line: " << line << "\n";
            return false;
        }
        points[i * 3 + 0] = x;
        points[i * 3 + 1] = y;
        points[i * 3 + 2] = z;
        normals[i * 3 + 0] = nx;
        normals[i * 3 + 1] = ny;
        normals[i * 3 + 2] = nz;
    }

    return true;
}

bool WriteAsciiPly(const std::string& path,
                   const std::vector<double>& vertices,
                   const std::vector<node_index_type>& faces) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to open output: " << path << "\n";
        return false;
    }

    const size_t v_count = vertices.size() / 3;
    const size_t f_count = faces.size() / 3;

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << v_count << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "element face " << f_count << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    for (size_t i = 0; i < v_count; ++i) {
        out << vertices[i * 3 + 0] << " "
            << vertices[i * 3 + 1] << " "
            << vertices[i * 3 + 2] << "\n";
    }

    for (size_t i = 0; i < f_count; ++i) {
        out << "3 "
            << static_cast<long long>(faces[i * 3 + 0]) << " "
            << static_cast<long long>(faces[i * 3 + 1]) << " "
            << static_cast<long long>(faces[i * 3 + 2]) << "\n";
    }

    return true;
}

bool WriteBinaryArray(const std::string& path, const std::vector<double>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output: " << path << "\n";
        return false;
    }
    uint64_t count = static_cast<uint64_t>(data.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()), sizeof(double) * data.size());
    }
    return true;
}

DumpOptions DefaultOptions() {
    DumpOptions options;
    options.params.depth = 8;
    options.params.full_depth = 5;
    options.params.cg_depth = 0;
    options.params.iters = 8;
    options.params.degree = 1;
    options.params.scale = 1.1;
    options.params.samples_per_node = 1.5;
    options.params.point_weight = 4.0;
    options.params.confidence = false;
    options.params.verbose = false;
    options.params.parallel_type = -1;
    options.params.grid_depth = 0;
    options.params.exact_interpolation = false;
    options.params.show_residual = false;
    options.params.low_depth_cutoff = 0.0;
    options.params.width = 0.0;
    options.params.cg_solver_accuracy = 1e-3;
    options.params.base_depth = -1;
    options.params.solve_depth = -1;
    options.params.kernel_depth = -1;
    options.params.base_v_cycles = 1;
    options.params.force_manifold = true;
    options.params.polygon_mesh = false;
    options.params.primal_grid = false;
    options.params.linear_fit = false;
    options.params.grid_coordinates = false;
    options.params.boundary_type = 2;
    options.params.dirichlet_erode = false;
    options.params.output_density = false;
    options.params.output_gradients = false;
    return options;
}

bool ParseArgs(int argc, char** argv, DumpOptions& options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--in" && i + 1 < argc) {
            options.input_ply = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            options.output_ply = argv[++i];
        } else if (arg == "--densityOut" && i + 1 < argc) {
            options.density_out = argv[++i];
        } else if (arg == "--gradientsOut" && i + 1 < argc) {
            options.gradients_out = argv[++i];
        } else if (arg == "--density") {
            options.params.output_density = true;
        } else if (arg == "--gradients") {
            options.params.output_gradients = true;
        } else if (arg == "--depth" && i + 1 < argc) {
            options.params.depth = std::stoi(argv[++i]);
        } else if (arg == "--fullDepth" && i + 1 < argc) {
            options.params.full_depth = std::stoi(argv[++i]);
        } else if (arg == "--cgDepth" && i + 1 < argc) {
            options.params.cg_depth = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            options.params.iters = std::stoi(argv[++i]);
        } else if (arg == "--degree" && i + 1 < argc) {
            options.params.degree = std::stoi(argv[++i]);
        } else if (arg == "--scale" && i + 1 < argc) {
            options.params.scale = std::stod(argv[++i]);
        } else if (arg == "--samplesPerNode" && i + 1 < argc) {
            options.params.samples_per_node = std::stod(argv[++i]);
        } else if (arg == "--pointWeight" && i + 1 < argc) {
            options.params.point_weight = std::stod(argv[++i]);
        } else if (arg == "--confidence") {
            options.params.confidence = true;
        } else if (arg == "--verbose") {
            options.params.verbose = true;
        } else if (arg == "--gridDepth" && i + 1 < argc) {
            options.params.grid_depth = std::stoi(argv[++i]);
        } else if (arg == "--exact") {
            options.params.exact_interpolation = true;
        } else if (arg == "--showResidual") {
            options.params.show_residual = true;
        } else if (arg == "--lowDepthCutOff" && i + 1 < argc) {
            options.params.low_depth_cutoff = std::stod(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            options.params.width = std::stod(argv[++i]);
        } else if (arg == "--cgAccuracy" && i + 1 < argc) {
            options.params.cg_solver_accuracy = std::stod(argv[++i]);
        } else if (arg == "--baseDepth" && i + 1 < argc) {
            options.params.base_depth = std::stoi(argv[++i]);
        } else if (arg == "--solveDepth" && i + 1 < argc) {
            options.params.solve_depth = std::stoi(argv[++i]);
        } else if (arg == "--kernelDepth" && i + 1 < argc) {
            options.params.kernel_depth = std::stoi(argv[++i]);
        } else if (arg == "--baseVCycles" && i + 1 < argc) {
            options.params.base_v_cycles = std::stoi(argv[++i]);
        } else if (arg == "--nonManifold") {
            options.params.force_manifold = false;
        } else if (arg == "--polygonMesh") {
            options.params.polygon_mesh = true;
        } else if (arg == "--primalGrid") {
            options.params.primal_grid = true;
        } else if (arg == "--linearFit") {
            options.params.linear_fit = true;
        } else if (arg == "--gridCoordinates") {
            options.params.grid_coordinates = true;
        } else if (arg == "--bType" && i + 1 < argc) {
            int val = std::stoi(argv[++i]);
            if (val == 1) {
                options.params.boundary_type = 0;
            } else if (val == 2) {
                options.params.boundary_type = 1;
            } else if (val == 3) {
                options.params.boundary_type = 2;
            } else {
                std::cerr << "Invalid --bType value: " << val << "\n";
                return false;
            }
        } else if (arg == "--dirichletErode") {
            options.params.dirichlet_erode = true;
        } else if (arg == "--noErode") {
            options.params.dirichlet_erode = false;
        } else if (arg == "--parallel" && i + 1 < argc) {
            options.params.parallel_type = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return false;
        }
    }

    if (options.input_ply.empty() || options.output_ply.empty()) {
        std::cerr << "--in and --out are required.\n";
        return false;
    }

    if (options.params.output_density && options.density_out.empty()) {
        options.density_out = DeriveOutputPath(options.output_ply, ".density.bin");
    }
    if (options.params.output_gradients && options.gradients_out.empty()) {
        options.gradients_out = DeriveOutputPath(options.output_ply, ".gradients.bin");
    }

    return true;
}

} // namespace

int main(int argc, char** argv) {
    DumpOptions options = DefaultOptions();
    if (!ParseArgs(argc, argv, options)) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::vector<double> points;
    std::vector<double> normals;
    if (!ReadAsciiPly(options.input_ply, points, normals)) {
        return 1;
    }

    const size_t count = points.size() / 3;
    void* mesh = run_poisson_reconstruction(points.data(), normals.data(), count, options.params);
    if (!mesh) {
        std::cerr << "Poisson reconstruction failed.\n";
        return 1;
    }

    const size_t vertex_count = get_vertex_count(mesh);
    const size_t face_count = get_face_count(mesh);

    std::vector<double> vertices(vertex_count * 3);
    std::vector<node_index_type> faces(face_count * 3);
    get_vertices_data(mesh, vertices.data());
    if (!faces.empty()) {
        get_faces_data(mesh, faces.data());
    }

    if (!WriteAsciiPly(options.output_ply, vertices, faces)) {
        free_mesh(mesh);
        return 1;
    }

    if (options.params.output_density) {
        std::vector<double> densities(get_density_count(mesh));
        if (!densities.empty()) {
            get_densities_data(mesh, densities.data());
        }
        if (!WriteBinaryArray(options.density_out, densities)) {
            free_mesh(mesh);
            return 1;
        }
    }

    if (options.params.output_gradients) {
        std::vector<double> gradients(get_gradient_count(mesh) * 3);
        if (!gradients.empty()) {
            get_gradients_data(mesh, gradients.data());
        }
        if (!WriteBinaryArray(options.gradients_out, gradients)) {
            free_mesh(mesh);
            return 1;
        }
    }

    if (options.params.verbose) {
        std::cout << "Iso-Value: " << get_iso_value(mesh) << "\n";
        std::cout << "Vertices: " << vertex_count << ", Faces: " << face_count << "\n";
    }

    free_mesh(mesh);
    return 0;
}
