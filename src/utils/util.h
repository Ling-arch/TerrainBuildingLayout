#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <Eigen/Dense>

namespace util
{
    using Eigen::Matrix2d, Eigen::Matrix2f;
    using Eigen::Vector2d;
    using std::vector, std::tuple, std::array;

    // 包含圆心和 3 个 Jacobian
    struct CircumcenterResult
    {
        public:
        Vector2d cc;                 // 圆心
        std::array<Matrix2d, 3> dcc; // 对 p0,p1,p2 的 Jacobian
    };

    std::tuple<Vector2d, Matrix2d, Matrix2d> dw_intersection_against_bisector(
        const Vector2d &ls, // line start
        const Vector2d &ld, // line dir
        const Vector2d &p0, // site0
        const Vector2d &p1  // site1
    );

    std::tuple<Vector2d, Matrix2d, Matrix2d> dw_intersection(
        const Vector2d &ls, // ray start
        const Vector2d &ld, // ray direction
        const Vector2d &a,  // bisector point
        const Vector2d &b   // bisector direction
    );

    Vector2d line_intersection(const Vector2d &ls, const Vector2d &ld,const Vector2d &ps, const Vector2d &pd);

    Vector2d circumcenter(const Vector2d &p0, const Vector2d &p1, const Vector2d &p2);

    CircumcenterResult wdw_circumcenter(const Vector2d &p0, const Vector2d &p1, const Vector2d &p2);

    inline Vector2d rotate90(const Vector2d &v) { return Vector2d(-v.y(), v.x()); }

    // polygon 几何：area,
    double polygon_area(const vector<Vector2d> &poly);

    //winding_number
    double winding_number(const vector<Vector2d> &poly, const Vector2d &p);

    // 将扁平数组转成 Vector2d 数组
    vector<Vector2d> to_vec2_array(const vector<double> &flat);

    vector<double> flat_vec2(const vector<Vector2d> &vec2arr);
}