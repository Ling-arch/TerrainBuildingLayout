#pragma once
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <set>
#include <map>
#include <algorithm>
#include <cassert>
#include <limits>
#include <tuple>
#include <functional>
#include <optional>
#include "util.h"

namespace voronoi2
{
    using std::array, std::map, std::set, std::vector, std::tuple;
    inline constexpr size_t INVALID_INDEX = static_cast<size_t>(std::numeric_limits<int64_t>::max());

    using Scalar = float;
    using M2 = util::Math2<Scalar>;
    using Vector2 = typename M2::Vector2;
    using Matrix2 = typename M2::Matrix2;

    struct Cell
    {
        vector<Vector2> vtx2xy; // 顶点坐标 (按顺序, ccw)

        // 多边形顶点：array[0] = loop_vertex_index,array[1] = INVALID_INDEX,array[2] = INVALID_INDEX,array[3] = INVALID_INDEX
        // 多边形边和两个site的垂直平分线的交点：array[0] = loop_vertex_index,array[1] = site1_index,array[2] = site2_index,array[3] = INVALID_INDEX
        // 三个site的外接圆心：array[0] = INVALID_INDEX,array[1] = site1_index,array[2] = site2_index,array[3] = site3_index
        vector<array<size_t, 4>> vtx2info; // 每个顶点的 info

        bool is_inside(const Vector2 &p) const;

        Scalar area() const;

    };

    Cell new_from_polyloop2(const vector<Vector2> &vtx2xy_in);

    Cell new_empty();

    // Voronoi Mesh 表示
    struct VoronoiMesh
    {
        vector<size_t> site2idx;            // prefix indices for idx2vtxv per site
        vector<size_t> idx2vtxv;            // flattened list of vertex indices per site
        vector<Vector2> vtxv2xy;            // coordinates of voronoi vertices
        vector<array<size_t, 4>> vtxv2info; // corresponding info per voronoi vertex
    };

    std::optional<Cell> hoge(
        const vector<Vector2> &vtx2xy,
        const vector<array<size_t, 4>> &vtx2info,
        const vector<tuple<Scalar, size_t, Vector2, array<size_t, 4>>> &vtxnews,
        const vector<size_t> &vtx2vtxnew,
        vector<char> &vtxnew2isvisited);

    // 高阶函数
    vector<Cell> cut_polygon_by_line(
        const Cell &cell,
        const Vector2 &line_s,
        const Vector2 &line_n,
        size_t i_vtx,
        size_t j_vtx);

    vector<Cell> voronoi_cells(
        const vector<Scalar> &vtxl2xy_flat, // flattened [x0,y0, x1,y1, ...] outer loop
        const vector<Scalar> &site2xy_flat, // flattened sites [x0,y0, ...]
        const std::function<bool(size_t)> &site2isalive);

    // 索引化：把每个 cell 的局部顶点去重映射到全局 voronoi 顶点
    VoronoiMesh indexing(const vector<Cell> &site2cell);

    // 根据 info 计算 voronoi vertex 的真实位置
    Vector2 position_of_voronoi_vertex(const array<size_t, 4> &info, const vector<Scalar> &vtxl2xy, const vector<Scalar> &site2xy);

}
