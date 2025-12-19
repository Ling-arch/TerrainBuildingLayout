#pragma once

#include "voronoi2.h"
#include "diffVoronoi.h"
#include "loss.h"
#include "polygonMesh.h"

namespace optimizer{
    inline constexpr size_t INVALID = static_cast<size_t>(std::numeric_limits<int64_t>::max());

    torch::Tensor optimize(
        std::vector<float> vtxl2xy,
        std::vector<float> site2xy_init,
        std::vector<size_t> site2room,
        std::vector<float> site2xy2flag,
        std::vector<float> room2area_trg,
        std::vector<std::pair<size_t, size_t>> room_connections);


    struct OptimizeProblem{

    };
}