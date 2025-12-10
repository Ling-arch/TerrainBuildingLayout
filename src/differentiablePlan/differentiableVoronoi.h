#pragma once


#include <iostream>
#include <vector>
#include <array>
#include "util.h"
#include "torch/torch.h"
#include <cstdint>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

using namespace util;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;


namespace diffVoronoi {

    struct VoronoiInfo{
        std::vector<int64_t> site2idx; // site2idx[site_id] = idx
        std::vector<int64_t> idx2vtxv; // idx2vtxv[idx] = vtxv_id
        std::vector<int64_t> idx2site; // idx2site[idx] = site_id
        std::vector<std::array<int64_t,4>> vtxv2info; //voronoi顶点的四种情况
    };

    class Voronoi2Layer : public torch::autograd::Function<Voronoi2Layer>
    {
        public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
            const torch::Tensor& site2xy,
            const std::vector<float>& vtxl2xy,
            const std::vector<std::array<int64_t,4>>& vtxv2info);
 
        static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);      
    };

    VoronoiInfo voronoi2(
        const torch::Tensor& site2xy,
        const std::vector<float>& vtxl2xy,
        const std::function<bool(int64_t)> &site2isalive);

    torch::Tensor loss_lloyd(
        const std::vector<int64_t>& elem2idx,
        const std::vector<int64_t>& idx2vtx,
        const torch::Tensor& site2xy,
        const torch::Tensor& vtxv2xy
    );

};

