#pragma once

#include <torch/torch.h>
#include <vector>
#include <array>
#include <cstdint>
#include <Eigen/Core>
#include "util.h"
#include "voronoi2.h"
#include "polygonMesh.h"
#include <iomanip>
#include <cmath>
#include <exception>
#include <random>

namespace diffVoronoi
{

    using std::array;
    using std::vector;
    using namespace torch::indexing;

    // using float to autograd
    using M2 = util::Math2<voronoi2::Scalar>;
    using Vector2 = typename M2::Vector2;
    using Matrix2 = typename M2::Matrix2;
    

    struct VoronoiInfo
    {
    public:
        vector<size_t> site2idx; // site index of site points
        vector<size_t> idx2vtxv; //
        vector<size_t> idx2site;
        vector<array<size_t, 4>> vtxv2info;
    };

    /*
     Voronoi autograd Function: forward(site2xy, vtxl2xy, vtxv2info) -> vtxv2xy
     - site2xy: Tensor (N_site, 2) float32
     - vtxl2xy: std::vector<double> flattened [x0,y0,x1,y1,...]
     - vtxv2info: std::vector<std::array<size_t,4>> (use size_t for safe torch IValue conversion)

     Implementation note:
     - Store site2xy in saved variables (tensor) and store vtxl2xy and vtxv2info in ctx->saved_data (IValue) for backward.
     - All geometric computations use double (Eigen::Vector2d). Conversion to/from tensors uses float32.
    */

    struct VoronoiParams
    {
        std::vector<float> vtxl2xy;                    // flattened loop coords
        std::vector<std::array<size_t, 4>> vtxv2info; // per-vertex info (use size_t)
    };

    class VoronoiFunction : public torch::autograd::Function<VoronoiFunction>
    {
    public:
        // forward: site2xy tensor + params passed as IValues (we wrap params into IValue-compatible types)
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                     const torch::Tensor &site2xy,
                                     const std::vector<float> &vtxl2xy,
                                     const std::vector<std::array<size_t, 4>> &vtxv2info);

        // backward: receives grad_output (grad wrt vtxv2xy)
        static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                                     torch::autograd::tensor_list grad_outputs);
    };

    // Convenience wrapper class similar to your earlier VoronoiLayer (holds params and calls Function)
    class VoronoiLayer
    {
    public:
        VoronoiLayer(const std::vector<float> &vtxl2xy_in,
                     const std::vector<std::array<size_t, 4>> &vtxv2info_in);

        // site2xy: (N,2) float32 -> returns (M,2) float32
        torch::Tensor forward(const torch::Tensor &site2xy) const;

        // helper to access params (if needed)
        const std::vector<float> &get_vtxl2xy() const { return vtxl2xy; }
        std::vector<std::array<size_t, 4>> get_vtxv2info_i64() const;

    private:
        std::vector<float> vtxl2xy;
        std::vector<std::array<size_t, 4>> vtxv2info;
    };

    std::pair<torch::Tensor, VoronoiInfo> voronoi(
        const std::vector<float> &vtxl2xy_f,             // flattened loop coords (f32)
        const torch::Tensor &site2xy,                    // Nx2 float32 tensor (CPU)
        const std::function<bool(size_t)> &site2isalive // predicate
    );

    // Forward: elem2idx (cumulative), idx2vtx (vertex indices), vtx2xy (num_vtx x 2 float tensor)
    // returns Tensor elem2cog (num_elem x 2 float)
    torch::Tensor polygonmesh2_to_cogs_forward(
        const std::vector<size_t> &elem2idx, // length = num_elem + 1, cumulative
        const std::vector<size_t> &idx2vtx,  // flattened vertex indices
        const torch::Tensor &vtx2xy           // num_vtx x 2 (float32, CPU)
    );

    // Backward: distribute dw_elem2cog (num_elem x 2) back to vertices (num_vtx x 2)
    // returns dw_vtx2xy tensor (num_vtx x 2)
    torch::Tensor polygonmesh2_to_cogs_backward(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const torch::Tensor &vtx2xy,     // used for shape (num_vtx,2)
        const torch::Tensor &dw_elem2cog // (num_elem,2)
    );

    torch::Tensor loss_lloyd(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const torch::Tensor &site2xy, // (num_sites,2) float32
        const torch::Tensor &vtxv2xy  // (num_vtxv,2) float32
    );

    // ---- Utility: convert Vector2 -> float32 tensor (N,2)
    std::vector<float> flat_tensor_to_float(const torch::Tensor &t);

    //convert float32 tensor row i to Eigen::Vector2
    inline Vector2 get_tensor_row_to_vec2(const float *ptr, size_t i);

    bool has_nan_inf(const torch::Tensor &t);

    void print_tensor_info(const std::string &name, const torch::Tensor &t);

    torch::Tensor vec2_to_tensor(const std::vector<Vector2> &list);

    void test_backward_cpp_exact(const std::vector<Vector2> &boundary,
                                 const std::vector<Vector2> &sites);
}
