#pragma once

#include "util.h"
#include <vector>
#include "voronoi2.h"
#include "diffVoronoi.h"
#include <cstdint>
#include <set>
#include <cassert>
#include <limits>
#include "polygonMesh.h"

namespace loss
{
    using diffVoronoi::PolygonMesh2ToCogsLayer, diffVoronoi::PolygonMesh2ToAreaLayer;
    using diffVoronoi::VoronoiInfo, diffVoronoi::PolygonMesh2ToCogsFunction, diffVoronoi::PolygonMesh2ToAreaFunction;
    using namespace polygonMesh;
    using Scalar = voronoi2::Scalar;
    using M2 = util::Math2<voronoi2::Scalar>;
    using Vector2 = typename M2::Vector2;

    inline constexpr size_t INVALID = static_cast<size_t>(std::numeric_limits<int64_t>::max());

    std::tuple<size_t, std::vector<size_t>, std::vector<std::vector<size_t>>> topology(
        const VoronoiInfo &voronoi_info,
        size_t num_room,
        const std::vector<size_t> &site2room);

    std::vector<std::vector<size_t>> inverse_map(size_t num_group, const std::vector<size_t> &site2group);

    bool is_two_room_connected(
        size_t i0_room,
        size_t i1_room,
        const std::vector<size_t> &site2room,
        const std::vector<std::vector<size_t>> &room2site,
        const VoronoiInfo &voronoi_info);

    std::pair<size_t, size_t> find_nearest_site(
        size_t i0_room,
        size_t i1_room,
        const std::vector<std::vector<size_t>> &room2site,
        const std::vector<float> &site2xy);

    torch::Tensor unidirectional(
        const torch::Tensor &site2xy,
        const std::vector<size_t> &site2room,
        size_t num_room,
        const VoronoiInfo &voronoi_info,
        const std::vector<std::pair<size_t, size_t>> &room_connections);

    std::vector<size_t> edge2vtvx_wall(
        const VoronoiInfo &voronoi_info,
        const std::vector<size_t> &site2room);

    torch::Tensor loss_lloyd_internal(
        const VoronoiInfo &voronoi_info,
        const std::vector<size_t> &site2room,
        const torch::Tensor &site2xy,
        const torch::Tensor &vtxv2xy);

    torch::Tensor room2area(
        const std::vector<size_t> &site2room,
        size_t num_room,
        const std::vector<size_t> &site2idx,
        const std::vector<size_t> &idx2vtxv,
        const torch::Tensor &vtxv2xy // (num_vtxv,2)
    );

    void remove_site_too_close(
        std::vector<size_t> &site2room,
        const torch::Tensor &site2xy // (num_site, 2)
    );

    std::vector<size_t> site2room(
        size_t num_site,
        const std::vector<float> &room2area);

    std::vector<size_t> site2room(
        size_t num_site,
        const std::vector<float> &room2area,
        const std::vector<size_t> &fixed_site_indices,
        const std::vector<size_t> &fixed_rooms);
        
    torch::Tensor loss_lloyd(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const torch::Tensor &site2xy, // (num_sites,2) float32
        const torch::Tensor &vtxv2xy  // (num_vtxv,2) float32
    );
}