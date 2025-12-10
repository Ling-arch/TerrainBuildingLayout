#pragma once


#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <Eigen/Dense>

using Eigen::Matrix2d;
using Eigen::Vector2d;

namespace util
{

    struct Vec2
    {
        float x, y;
        Vec2(float _x = 0.0f, float _y = 0.0f) : x(_x), y(_y) {}
        Vec2 operator+(const Vec2 &other) const { return Vec2(x + other.x, y + other.y); }
        Vec2 operator-(const Vec2 &other) const { return Vec2(x - other.x, y - other.y); }
        Vec2 operator*(float scalar) const { return Vec2(x * scalar, y * scalar); }
        float dot(const Vec2 &other) const { return x * other.x + y * other.y; }
        float length() const { return std::sqrt(x * x + y * y); }
        Vec2 normalized() const
        {
            float len = length();
            if (len == 0)
                return Vec2(0, 0);
            return Vec2(x / len, y / len);
        }
    };

    // 从vector读取转换成Vec2
    inline Vec2 to_vec2(const std::vector<float> &data, int64_t idx)
    {
        return Vec2(data[2 * idx], data[2 * idx + 1]);
    }

    // 求三个点的外接圆中心
    std::tuple<Vec2, std::array<Vec2, 3>> circumcenter(const Vec2 &s0, const Vec2 &s1, const Vec2 &s2);

    // 求直线与中垂线的交点
    std::tuple<Vec2, Vec2, Vec2> dw_intersection_against_bisector(const Vec2 &l1, const Vec2 &ldir,
                                                                  const Vec2 &s0, const Vec2 &s1);

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

    // 计算Voronoi顶点位置
    Vec2 compute_voronoi_vertex(const std::array<int64_t, 4> &info,
                                    const std::vector<float> &vtxl2xy,
                                    const std::vector<float> &site2xy);


    inline Vector2d rotate90(const Vector2d &v){return Vector2d(-v.y(), v.x());}

};

