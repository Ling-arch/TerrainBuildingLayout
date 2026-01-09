#pragma once

#include <cassert>
#include "util.h"


namespace polyloop
{

    using Scalar = float;
    using M2 = util::Math2<Scalar>;
    using Vector2 = typename M2::Vector2;
    using Vector3 = typename M2::Vector3;
    using Matrix2 = typename M2::Matrix2;
    using Tri = util::Tri;
    
    struct AABB2
    {
        float minx, miny;
        float maxx, maxy;
    };

    template <typename Real>
    struct NormalizeTransform2D
    {
        Real cx, cy; // 原始 AABB center
        Real scale;  // size / max_edge_size
        Real tx, ty; // center_pos
    };

    template <typename T>
    AABB2 aabb2(const std::vector<T> &vtx2xy)
    {
        assert(vtx2xy.size() % 2 == 0);

        AABB2 aabb;
        aabb.minx = aabb.maxx = vtx2xy[0];
        aabb.miny = aabb.maxy = vtx2xy[1];

        for (size_t i = 0; i < vtx2xy.size(); i += 2)
        {
            aabb.minx = std::min(aabb.minx, vtx2xy[i]);
            aabb.miny = std::min(aabb.miny, vtx2xy[i + 1]);
            aabb.maxx = std::max(aabb.maxx, vtx2xy[i]);
            aabb.maxy = std::max(aabb.maxy, vtx2xy[i + 1]);
        }
        return aabb;
    }

    template <typename T>
    inline void aabb_center(const AABB2 &aabb, T &cx, T &cy)
    {
        cx = (aabb.minx + aabb.maxx) * T(0.5);
        cy = (aabb.miny + aabb.maxy) * T(0.5);
    }

    template <typename T>
    inline T aabb_max_edge_size(const AABB2 &aabb)
    {
        return std::max(
            T(aabb.maxx - aabb.minx),
            T(aabb.maxy - aabb.miny));
    }

    class Polyloop2
    {
    public:
        Polyloop2(const std::vector<Vector2> &points);
      

        const std::vector<Vector2> &points() const { return points_; }

        const std::vector<Tri> &triangles() const { return triangles_; }

        const double &get_area() { return double(M2::polygon_area(points_)); }

    private:
        std::vector<Vector2> points_;
        std::vector<Tri> triangles_;
    };

    class Polyloop3
    {
    public:
        Polyloop3(const std::vector<Vector3> &points);
       
        const std::vector<Vector3> &points() const { return points_; }
        const std::vector<Vector2> &projected_points() const { return projected_points_; }
        const std::vector<Tri> &triangles() const { return triangles_; }
        const Vector3 &normal() const { return normal_; }

    private:
        std::vector<Vector3> points_;
        Vector3 normal_, u_, v_;
        std::vector<Vector2> projected_points_;
        std::vector<Tri> triangles_;
    };

    template <typename T, int N>
    T arclength(const std::vector<T> &vtx2xyz)
    {
        static_assert(N > 0);
        assert(vtx2xyz.size() % N == 0);

        const size_t np = vtx2xyz.size() / N;
        T len = T(0);

        for (size_t ip0 = 0; ip0 < np; ++ip0)
        {
            size_t ip1 = (ip0 + 1) % np;

            const T *p0 = &vtx2xyz[ip0 * N];
            const T *p1 = &vtx2xyz[ip1 * N];

            T dist2 = T(0);
            for (int d = 0; d < N; ++d)
            {
                T diff = p1[d] - p0[d];
                dist2 += diff * diff;
            }
            len += std::sqrt(dist2);
        }
        return len;
    }

    //T is float/double, N is demension of points coordinates
    template <typename T, int N>
    std::vector<T> resample(const std::vector<T> &vtx2xyz_in, size_t num_edge_out)
    {
        static_assert(N > 0);
        assert(vtx2xyz_in.size() % N == 0);

        const size_t num_edge_in = vtx2xyz_in.size() / N;

        std::vector<T> vtx_out;
        vtx_out.reserve(num_edge_out * N);

        // total length / target edge count
        const T total_len = arclength<T, N>(vtx2xyz_in);
        const T len_edge_out = total_len / T(num_edge_out);

        // push first vertex
        for (int d = 0; d < N; ++d)
            vtx_out.push_back(vtx2xyz_in[d]);

        size_t i_edge_in = 0;
        T traveled_ratio0 = T(0);
        T remaining_length = len_edge_out;

        while (true)
        {
            if (i_edge_in >= num_edge_in)
                break;
            if (vtx_out.size() >= num_edge_out * N)
                break;

            const size_t i0 = i_edge_in;
            const size_t i1 = (i_edge_in + 1) % num_edge_in;

            const T *p0 = &vtx2xyz_in[i0 * N];
            const T *p1 = &vtx2xyz_in[i1 * N];

            // compute edge length
            T len_edge0 = T(0);
            for (int d = 0; d < N; ++d)
            {
                T diff = p1[d] - p0[d];
                len_edge0 += diff * diff;
            }
            len_edge0 = std::sqrt(len_edge0);

            const T len_togo0 = len_edge0 * (T(1) - traveled_ratio0);

            if (len_togo0 > remaining_length)
            {
                // insert point inside this edge
                traveled_ratio0 += remaining_length / len_edge0;

                for (int d = 0; d < N; ++d)
                {
                    T v = p0[d] * (T(1) - traveled_ratio0) + p1[d] * traveled_ratio0;
                    vtx_out.push_back(v);
                }
                remaining_length = len_edge_out;
            }
            else
            {
                // move to next edge
                remaining_length -= len_togo0;
                traveled_ratio0 = T(0);
                ++i_edge_in;
            }
        }

        return vtx_out;
    }

    template <typename Real>
    std::vector<Real> normalize(
        const std::vector<Real> &vtx2xy,
        const std::array<Real, 2> &center_pos,
        Real size,
        NormalizeTransform2D<Real> &out_tf)
    {
        assert(vtx2xy.size() % 2 == 0);

        AABB2 aabb = aabb2(vtx2xy);

        Real cx, cy;
        aabb_center(aabb, cx, cy);

        Real max_edge_size = aabb_max_edge_size<Real>(aabb);
        Real scale = size / max_edge_size;

        // 记录 transform
        out_tf.cx = cx;
        out_tf.cy = cy;
        out_tf.scale = scale;
        out_tf.tx = center_pos[0];
        out_tf.ty = center_pos[1];

        std::vector<Real> out;
        out.reserve(vtx2xy.size());

        for (size_t i = 0; i < vtx2xy.size(); i += 2)
        {
            out.push_back((vtx2xy[i] - cx) * scale + center_pos[0]);
            out.push_back((vtx2xy[i + 1] - cy) * scale + center_pos[1]);
        }
        return out;
    }


    template <typename Real>
    std::vector<Real> map_pt_normalized(const std::vector<Real> &pt2xy, const std::array<Real, 2> &center_pos, const NormalizeTransform2D<Real>& tf)
    {
        assert(pt2xy.size() % 2 == 0);
        std::vector<Real> out;
        for (size_t i = 0; i < pt2xy.size(); i += 2)
        {
            out.push_back((pt2xy[i] - tf.cx) * tf.scale + center_pos[0]);
            out.push_back((pt2xy[i + 1] - tf.cy) * tf.scale + center_pos[1]);
        }
        return out;
    }

    inline std::vector<Vector2> map_pt_normalized(const std::vector<Vector2> &pts, const std::array<Scalar, 2> &center_pos, const NormalizeTransform2D<Scalar> &tf)
    {
        std::vector<Vector2> out;
        for (Vector2 v : pts)
        {
            Vector2 mapped_pt = {(v.x() - tf.cx) * tf.scale + center_pos[0], (v.y() - tf.cy) * tf.scale + center_pos[1]};
            out.push_back(mapped_pt);
        }
        return out;
    }

    template <typename Real>
    std::vector<Real> denormalize(
        const std::vector<Real> &vtx2xy_norm,
        const NormalizeTransform2D<Real> &tf)
    {
        assert(vtx2xy_norm.size() % 2 == 0);

        std::vector<Real> out;
        out.reserve(vtx2xy_norm.size());

        for (size_t i = 0; i < vtx2xy_norm.size(); i += 2)
        {
            Real x = (vtx2xy_norm[i] - tf.tx) / tf.scale + tf.cx;
            Real y = (vtx2xy_norm[i + 1] - tf.ty) / tf.scale + tf.cy;
            out.push_back(x);
            out.push_back(y);
        }
        return out;
    }

    inline std::vector<Vector2> denormalize(
        const std::vector<Vector2> &vtx_norm,
        const NormalizeTransform2D<Scalar> &tf)
    {

        std::vector<Vector2> out;
        out.reserve(vtx_norm.size());

        for (Vector2 v : vtx_norm)
        {
            Vector2 v_origin = {(v.x() - tf.tx) / tf.scale + tf.cx, (v.y() - tf.ty) / tf.scale + tf.cy};
            out.push_back(v_origin);
        }
        return out;
    }

    inline std::vector<Polyloop3> convert_polyloop2_to_3d(std::vector<Polyloop2> polys, Scalar z)
    {
        std::vector<Polyloop3> out;
        for (Polyloop2 &poly : polys)
        {
            std::vector<Vector3> points;
            for (Vector2 pt : poly.points())
            {
                points.push_back({pt.x(), pt.y(), z});
            }
            out.emplace_back(points);
        }
        return out;
    }

    inline Polyloop3 convert_points_to_polyloop3d(std::vector<Vector2> points, Scalar z){
        std::vector<Vector3> points_3d;
        for (Vector2 pt : points)
        {
            points_3d.push_back({pt.x(), pt.y(), z});
        }
        return Polyloop3(points_3d);
    }

    inline std::vector<Vector3> convert_points_to_3d(std::vector<Vector2> points, Scalar z)
    {
        std::vector<Vector3> points_3d;
        for (Vector2 pt : points)
        {
            points_3d.push_back({pt.x(), pt.y(), z});
        }
        return points_3d;
    }

    inline std::vector<Vector3> denormalize_and_to3d(const std::vector<Vector2> &vtx_norm, const NormalizeTransform2D<Scalar> &tf, Scalar z)
    {
        std::vector<Vector3> out;
        out.reserve(vtx_norm.size());

        for (Vector2 v : vtx_norm)
        {
            Vector3 v_origin = {(v.x() - tf.tx) / tf.scale + tf.cx, (v.y() - tf.ty) / tf.scale + tf.cy, z};
            out.push_back(v_origin);
        }
        return out;
    }
}