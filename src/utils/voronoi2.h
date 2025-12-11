#pragma once
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <set>
#include <map>
#include <functional>
#include <optional>
#include "util.h"

using Eigen::Matrix2d,Eigen::Vector2d;
using std::array,std::map,std::set,std::size_t,std::vector;

namespace voronoi2
{

    // ---- 类型 ----
    struct Cell
    {
        vector<Vector2d> vtx2xy;           // 顶点坐标 (按顺序, ccw)
        vector<array<size_t, 4>> vtx2info; // 每个顶点的 info (对应 Rust 的 [usize;4])

        // 简单方法
        bool is_inside(const Vector2d &p) const;
        double area() const;

        // 构造
        static Cell new_from_polyloop2(const vector<Vector2d> &vtx2xy_in);
        static Cell new_empty();
    };

    // Voronoi Mesh 表示
    struct VoronoiMesh
    {
        vector<size_t> site2idx;            // prefix indices for idx2vtxv per site
        vector<size_t> idx2vtxv;            // flattened list of vertex indices per site
        vector<Vector2d> vtxv2xy;           // coordinates of voronoi vertices
        vector<array<size_t, 4>> vtxv2info; // corresponding info per voronoi vertex
    };

    // 高阶函数
    vector<Cell> cut_polygon_by_line(
        const Cell &cell,
        const Vector2d &line_s,
        const Vector2d &line_n,
        size_t i_vtx,
        size_t j_vtx);

    vector<Cell> voronoi_cells(
        const vector<double> &vtxl2xy, // flattened [x0,y0, x1,y1, ...] outer loop
        const vector<double> &site2xy, // flattened sites [x0,y0, ...]
        const std::function<bool(size_t)> &site2isalive);

    // 索引化：把每个 cell 的局部顶点去重映射到全局 voronoi 顶点
    VoronoiMesh indexing(const vector<Cell> &site2cell);

    // 根据 info 计算 voronoi vertex 的真实位置（与 Rust 中 position_of_voronoi_vertex 对应）
    Vector2d position_of_voronoi_vertex(
        const array<size_t, 4> &info,
        const vector<double> &vtxl2xy,
        const vector<double> &site2xy);

        
    // ---- 辅助工具（对外可见） ----
    Vector2d line_intersection_param(
        const Vector2d &ls, const Vector2d &ld,
        const Vector2d &qs, const Vector2d &qd); // 返回交点（ls + t*ld）

    Vector2d circumcenter(const Vector2d &a, const Vector2d &b, const Vector2d &c);

    // 将扁平数组转成 Vector2d 数组
    vector<Vector2d> to_vec2_array(const vector<double> &flat);

    // polygon 几何：area, winding_number
    double polygon_area(const vector<Vector2d> &poly);
    double winding_number(const vector<Vector2d> &poly, const Vector2d &p);

}
