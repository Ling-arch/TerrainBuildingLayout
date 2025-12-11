#pragma once

#include <torch/torch.h>
#include <vector>
#include <array>
#include <cstdint>
#include <Eigen/Core>
#include "util.h"
#include "voronoi2.h"

namespace diffVoronoi
{

    using Eigen::Matrix2d;
    using Eigen::Vector2d;
    using std::array;
    using std::vector;

    struct VoronoiInfo{
    public:
        vector<size_t> site2idx;           //site index of site points
        vector<size_t> idx2vtxv;           //
        vector<size_t> idx2site;
        vector<array<size_t,4>> vtxv2info;

    };

    /*
     Voronoi autograd Function: forward(site2xy, vtxl2xy, vtxv2info) -> vtxv2xy
     - site2xy: Tensor (N_site, 2) float32
     - vtxl2xy: std::vector<double> flattened [x0,y0,x1,y1,...]
     - vtxv2info: std::vector<std::array<int64_t,4>> (use int64_t for safe torch IValue conversion)

     Implementation note:
     - Store site2xy in saved variables (tensor) and store vtxl2xy and vtxv2info in ctx->saved_data (IValue) for backward.
     - All geometric computations use double (Eigen::Vector2d). Conversion to/from tensors uses float32.
    */

    struct VoronoiParams
    {
        std::vector<double> vtxl2xy;                   // flattened loop coords
        std::vector<std::array<size_t, 4>> vtxv2info; // per-vertex info (use size_t)
    };

    class VoronoiFunction : public torch::autograd::Function<VoronoiFunction>{
    public:
        // forward: site2xy tensor + params passed as IValues (we wrap params into IValue-compatible types)
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                     const torch::Tensor &site2xy,
                                     const std::vector<double> &vtxl2xy,
                                     const std::vector<std::array<size_t, 4>> &vtxv2info);

        // backward: receives grad_output (grad wrt vtxv2xy)
        static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                              torch::autograd::tensor_list grad_outputs);
    };

    // Convenience wrapper class similar to your earlier VoronoiLayer (holds params and calls Function)
    class VoronoiLayer{
    public:
        VoronoiLayer(const std::vector<double> &vtxl2xy_in,
                     const std::vector<std::array<size_t, 4>> &vtxv2info_in);

        // site2xy: (N,2) float32 -> returns (M,2) float32
        torch::Tensor forward(const torch::Tensor &site2xy) const;

        // helper to access params (if needed)
        const std::vector<double> &get_vtxl2xy() const { return vtxl2xy; }
        std::vector<std::array<size_t, 4>> get_vtxv2info_i64() const;

    private:
        std::vector<double> vtxl2xy;
        std::vector<std::array<size_t, 4>> vtxv2info;
    };

}
