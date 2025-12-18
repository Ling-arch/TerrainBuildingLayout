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
                     const std::vector<std::array<size_t, 4>> &vtxv2info_in)
            : vtxl2xy(vtxl2xy_in), vtxv2info(vtxv2info_in) {};

        // site2xy: (N,2) float32 -> returns (M,2) float32
        torch::Tensor forward(const torch::Tensor &site2xy) const
        {
            // call autograd::Function apply
            // note: torch::autograd::Function::apply signature expects the same types we defined for forward
            return VoronoiFunction::apply(site2xy, vtxl2xy, vtxv2info);
        };

        // helper to access params
        const std::vector<float> &get_vtxl2xy() const { return vtxl2xy; };
        std::vector<std::array<size_t, 4>> get_vtxv2info_i64() const { return vtxv2info; };

    private:
        std::vector<float> vtxl2xy;
        std::vector<std::array<size_t, 4>> vtxv2info;
    };

    std::pair<torch::Tensor, VoronoiInfo> voronoi(
        const std::vector<float> &vtxl2xy_f,            // flattened loop coords (f32)
        const torch::Tensor &site2xy,                   // Nx2 float32 tensor (CPU)
        const std::function<bool(size_t)> &site2isalive // predicate
    );

    class PolygonMesh2ToCogsFunction : public torch::autograd::Function<PolygonMesh2ToCogsFunction>
    {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext *ctx,
            const torch::Tensor &vtx2xy,
            const std::vector<size_t> &elem2idx,
            const std::vector<size_t> &idx2vtx);

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::tensor_list grad_outputs);
    };

    class PolygonMesh2ToCogsLayer
    {
    public:
        PolygonMesh2ToCogsLayer(const std::vector<size_t> &elem2idx_,
                                const std::vector<size_t> &idx2vtx_)
            : elem2idx(elem2idx_), idx2vtx(idx2vtx_) {}

        // site2xy: (N,2) float32 -> returns (M,2) float32
        torch::Tensor forward(const torch::Tensor &vtx2xy) const
        {
            return PolygonMesh2ToCogsFunction::apply(vtx2xy, elem2idx, idx2vtx);
        }

        // helper to access params (if needed)
        const std::vector<size_t> &get_elem2idx() const { return elem2idx; }
        const std::vector<size_t> &get_idx2vtx() const { return idx2vtx; }

    private:
        std::vector<size_t> elem2idx;
        std::vector<size_t> idx2vtx;
    };

    class PolygonMesh2ToAreaFunction : public torch::autograd::Function<PolygonMesh2ToAreaFunction>
    {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext *ctx,
            const torch::Tensor &vtx2xy,
            const std::vector<size_t> &elem2idx,
            const std::vector<size_t> &idx2vtx);

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::tensor_list grad_outputs);
    };

    class PolygonMesh2ToAreaLayer
    {
    public:
        PolygonMesh2ToAreaLayer(const std::vector<size_t> &elem2idx_, const std::vector<size_t> &idx2vtx_)
            : elem2idx(elem2idx_), idx2vtx(idx2vtx_) {}

        torch::Tensor forward(const torch::Tensor &vtx2xy) const
        {
            return PolygonMesh2ToAreaFunction::apply(vtx2xy, elem2idx, idx2vtx);
        }

        const std::vector<size_t> &get_elem2idx() const { return elem2idx; }
        const std::vector<size_t> &get_idx2vtx() const { return idx2vtx; }

    private:
        std::vector<size_t> elem2idx;
        std::vector<size_t> idx2vtx;
    };

    class Vtx2XYZToEdgeVectorFunction : public torch::autograd::Function<Vtx2XYZToEdgeVectorFunction>
    {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext *ctx,
            const torch::Tensor &vtx2xy,        // (num_vtx, num_dim)
            const std::vector<size_t> &edge2vtx // len = num_edge * 2
        );

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::tensor_list grad_outputs);
    };

    class Vtx2XYZToEdgeVectorLayer
    {
    public:
        Vtx2XYZToEdgeVectorLayer(std::vector<size_t> edge2vtx_)
            : edge2vtx(edge2vtx_) {}

        torch::Tensor forward(const torch::Tensor &vtx2xy) const
        {
            return Vtx2XYZToEdgeVectorFunction::apply(vtx2xy, edge2vtx);
        }

        const std::vector<size_t> &get_egde2vtx() const { return edge2vtx; }

    private:
        std::vector<size_t> edge2vtx;
    };

    // ---- Utility: convert Vector2 -> float32 tensor (N,2)
    std::vector<float> flat_tensor_to_float(const torch::Tensor &t);

    // convert float32 tensor row i to Eigen::Vector2
    inline Vector2 get_tensor_row_to_vec2(const float *ptr, size_t i);

    bool has_nan_inf(const torch::Tensor &t);

    void print_tensor_info(const std::string &name, const torch::Tensor &t);

    torch::Tensor vec2_to_tensor(const std::vector<Vector2> &list);

    void test_backward_cpp_exact(const std::vector<Vector2> &boundary,
                                 const std::vector<Vector2> &sites);
}
