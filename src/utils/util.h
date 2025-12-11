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

    CircumcenterResult wdw_circumcenter(
        const Vector2d &p0,
        const Vector2d &p1,
        const Vector2d &p2);

    inline Vector2d rotate90(const Vector2d &v) { return Vector2d(-v.y(), v.x()); }

    inline double norm_squared(const Vector2d &v) { return v.squaredNorm(); }
}