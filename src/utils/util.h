#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <tuple>
#include <random>
#include <Eigen/Dense>
#include <array>
#include <nanoflann.hpp>
#include <earcut.hpp>

namespace util
{
    using std::vector, std::tuple, std::array;
    constexpr double Litten_PI = 3.14159265358979323846;
    using Tri = std::array<size_t, 3>;
    template <typename Scalar>
    struct Math2
    {
        using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
        using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
        using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

        // 包含圆心和 3 个 Jacobian
        struct CircumcenterResult
        {
        public:
            Vector2 cc;            // 圆心
            array<Matrix2, 3> dcc; // 对 p0,p1,p2 的 Jacobian
        };

        static tuple<Vector2, Matrix2, Matrix2> dw_intersection_against_bisector(
            const Vector2 &ls, // line start
            const Vector2 &ld, // line dir
            const Vector2 &p0, // site0
            const Vector2 &p1  // site1
        )
        {
            const Scalar half = Scalar(0.5);

            // middle point
            Vector2 a = (p0 + p1) * half;
            Vector2 b = rotate90(p1 - p0);

            Vector2 r;
            Matrix2 drda, drdb;
            std::tie(r, drda, drdb) = dw_intersection(ls, ld, a, b);

            Matrix2 dbdp0;
            dbdp0 << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);

            Matrix2 dbdp1;
            dbdp1 << Scalar(0), Scalar(-1), Scalar(1), Scalar(0);

            // chain rule
            Matrix2 drdp0 = drda * half + drdb * dbdp0;
            Matrix2 drdp1 = drda * half + drdb * dbdp1;

            return {r, drdp0, drdp1};
        }

        static Vector2 circumcenter(const Vector2 &p0, const Vector2 &p1, const Vector2 &p2)
        {
            Scalar a0 = (p1 - p2).squaredNorm();
            Scalar a1 = (p2 - p0).squaredNorm();
            Scalar a2 = (p0 - p1).squaredNorm();
            Scalar b0 = a0 * (a1 + a2 - a0);
            Scalar b1 = a1 * (a0 + a2 - a1);
            Scalar b2 = a2 * (a0 + a1 - a2);
            Scalar sum = b0 + b1 + b2;
            Scalar sum_inv = Scalar(1) / sum;
            Scalar c0 = b0 * sum_inv;
            Scalar c1 = b1 * sum_inv;
            Scalar c2 = b2 * sum_inv;
            return c0 * p0 + c1 * p1 + c2 * p2;
        }

        static tuple<Vector2, Matrix2, Matrix2> dw_intersection(
            const Vector2 &ps, // point on ray 1
            const Vector2 &pd, // direction of ray 1
            const Vector2 &qs, // point on ray 2
            const Vector2 &qd  // direction of ray 2
        )
        {
            Vector2 qn = rotate90(qd);
            Vector2 a = qs - ps;

            Scalar denom = pd.dot(qn);
            Scalar b = Scalar(1) / denom;
            Scalar t = a.dot(qn) * b;
            Vector2 r = ps + pd * t;
            Vector2 dt_dqn = a * b - pd * (a.dot(qn) * b * b);
            Vector2 dt_dqd = rotate90(dt_dqn);
            Vector2 dt_dqs = qn * b;

            Matrix2 dr_dqs = pd * dt_dqs.transpose();
            Matrix2 dr_dqd = pd * dt_dqd.transpose();
            // 返回交点r和关于qs和qd的导数矩阵
            return {r, dr_dqs, dr_dqd};
        }

        static Vector2 line_intersection(
            const Vector2 &ls,
            const Vector2 &ld,
            const Vector2 &ps,
            const Vector2 &pd)
        {
            Vector2 qn = rotate90(pd);
            Vector2 a = ps - ls;

            Scalar denom = ld.dot(qn);
            Scalar b = Scalar(1) / denom;
            Scalar t = a.dot(qn) * b;
            return ls + ld * t;
        }

        static CircumcenterResult wdw_circumcenter(
            const Vector2 &p0,
            const Vector2 &p1,
            const Vector2 &p2)
        {
            // 1) 计算边长平方
            Scalar a0 = (p1 - p2).squaredNorm();
            Scalar a1 = (p2 - p0).squaredNorm();
            Scalar a2 = (p0 - p1).squaredNorm();

            // 2) 计算权重 b0,b1,b2
            Scalar b0 = a0 * (a1 + a2 - a0);
            Scalar b1 = a1 * (a2 + a0 - a1);
            Scalar b2 = a2 * (a0 + a1 - a2);

            Scalar sum = b0 + b1 + b2;
            Scalar sum_inv = Scalar(1.0) / sum;

            // 归一化权重
            Scalar c0 = b0 * sum_inv;
            Scalar c1 = b1 * sum_inv;
            Scalar c2 = b2 * sum_inv;

            // 3) 圆心
            Vector2 cc = c0 * p0 + c1 * p1 + c2 * p2;

            // 4) db0, db1, db2 计算
            Scalar two = Scalar(2);

            array<Vector2, 3> db0 = {
                (p0 - p2 + p0 - p1) * (two * a0),
                (p1 - p0 + p2 - p1) * (two * a0) + (p1 - p2) * (two * (a1 + a2 - a0)),
                (p2 - p0 + p1 - p2) * (two * a0) + (p2 - p1) * (two * (a1 + a2 - a0)),
            };

            array<Vector2, 3> db1 = {
                (p0 - p1 + p2 - p0) * (two * a1) + (p0 - p2) * (two * (a2 + a0 - a1)),
                (p1 - p0 + p1 - p2) * (two * a1),
                (p2 - p1 + p0 - p2) * (two * a1) + (p2 - p0) * (two * (a2 + a0 - a1)),
            };

            array<Vector2, 3> db2 = {
                (p0 - p2 + p1 - p0) * (two * a2) + (p0 - p1) * (two * (a0 + a1 - a2)),
                (p1 - p2 + p0 - p1) * (two * a2) + (p1 - p0) * (two * (a0 + a1 - a2)),
                (p2 - p1 + p2 - p0) * (two * a2),
            };

            // 5) d(sum_inv)/dpi
            Scalar tmp = -Scalar(1) / (sum * sum);

            array<Vector2, 3> dsum_inv = {
                (db0[0] + db1[0] + db2[0]) * tmp,
                (db0[1] + db1[1] + db2[1]) * tmp,
                (db0[2] + db1[2] + db2[2]) * tmp,
            };

            // 6) 最终的 dcc[i] = dcc/dpi
            array<Matrix2, 3> dcc;

            // i = 0
            dcc[0] =
                c0 * Matrix2::Identity() +
                p0 * (db0[0] * sum_inv + dsum_inv[0] * b0).transpose() +
                p1 * (db1[0] * sum_inv + dsum_inv[0] * b1).transpose() +
                p2 * (db2[0] * sum_inv + dsum_inv[0] * b2).transpose();

            // i = 1
            dcc[1] =
                c1 * Matrix2::Identity() +
                p0 * (db0[1] * sum_inv + dsum_inv[1] * b0).transpose() +
                p1 * (db1[1] * sum_inv + dsum_inv[1] * b1).transpose() +
                p2 * (db2[1] * sum_inv + dsum_inv[1] * b2).transpose();

            // i = 2
            dcc[2] =
                c2 * Matrix2::Identity() +
                p0 * (db0[2] * sum_inv + dsum_inv[2] * b0).transpose() +
                p1 * (db1[2] * sum_inv + dsum_inv[2] * b1).transpose() +
                p2 * (db2[2] * sum_inv + dsum_inv[2] * b2).transpose();

            return {cc, dcc};
        }

        static Scalar polygon_area(const vector<Vector2> &poly)
        {
            return Eigen::numext::abs(signed_polygon_area(poly));
        }

        static Scalar signed_polygon_area(const vector<Vector2> &poly)
        {
            if (poly.size() < 3)
                return Scalar(0);
            Scalar s = Scalar(0);
            for (int i = 0; i < poly.size(); ++i)
            {
                int j = (i + 1) % poly.size();
                s += poly[i].x() * poly[j].y() - poly[j].x() * poly[i].y();
            }
            return Scalar(0.5) * s;
        }

        // Winding number (robust-ish) for point-in-polygon
        static Scalar winding_number(const vector<Vector2> &poly, const Vector2 &p)
        {
            int wn = 0;
            for (size_t i = 0; i < poly.size(); ++i)
            {
                Vector2 a = poly[i];
                Vector2 b = poly[(i + 1) % poly.size()];

                Vector2 ab = b - a;
                Vector2 ap = p - a;
                Scalar isLeft = ab.x() * ap.y() - ab.y() * ap.x();
                if (a.y() <= p.y())
                {
                    if (b.y() > p.y())
                    {
                        if (isLeft > 0)
                            ++wn;
                    }
                }
                else
                {
                    if (b.y() <= p.y())
                    {
                        if (isLeft < 0)
                            --wn;
                    }
                }
            }

            return static_cast<Scalar>(wn);
        }

        static inline vector<Vector2> to_vec2_array(const vector<Scalar> &flat)
        {
            assert(flat.size() % 2 == 0);
            vector<Vector2> out;
            out.reserve(flat.size() / 2);
            for (int i = 0; i + 1 < flat.size(); i += 2)
                out.emplace_back(flat[i], flat[i + 1]);
            return out;
        }

        static inline vector<Scalar> flat_vec2(const vector<Vector2> &vec2arr)
        {
            vector<Scalar> out;
            out.reserve(vec2arr.size() * 2);
            for (const auto &p : vec2arr)
            {
                out.push_back(p.x());
                out.push_back(p.y());
            }
            return out;
        }

        static inline Vector2 rotate90(const Vector2 &v) { return Vector2(-v.y(), v.x()); }

        static std::vector<Vector2> gen_random_sites(const std::vector<Vector2> &poly, size_t n, Scalar minx, Scalar maxx, Scalar miny, Scalar maxy, unsigned seed = 0)
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<Scalar> dx(minx, maxx);
            std::uniform_real_distribution<Scalar> dy(miny, maxy);

            std::vector<Vector2> sites;
            sites.reserve(n);

            size_t count = 0;
            size_t try_nums = 0;

            while (count < n && try_nums < 1000)
            {
                Vector2 p;
                p << dx(rng), dy(rng);

                if (std::abs(winding_number(poly, p) - Scalar(1)) < Scalar(0.1))
                {
                    sites.push_back(p);
                    ++count;
                }
                ++try_nums;
            }
            return sites;
        }

        static bool point_in_poly(const std::vector<Vector2> &poly, const Vector2 &p)
        {
            return std::abs(winding_number(poly, p) - Scalar(1)) < Scalar(0.1);
        }

        static std::vector<Vector2> gen_poisson_sites_in_poly(
            const std::vector<Vector2> &poly,
            Scalar r,
            int k = 30,
            unsigned seed = 0)
        {
            std::mt19937_64 rng(seed);

            // ---------- bounding box ----------
            Scalar minx = poly[0].x(), maxx = poly[0].x();
            Scalar miny = poly[0].y(), maxy = poly[0].y();
            for (auto &v : poly)
            {
                minx = std::min(minx, v.x());
                maxx = std::max(maxx, v.x());
                miny = std::min(miny, v.y());
                maxy = std::max(maxy, v.y());
            }

            std::uniform_real_distribution<Scalar> dx(minx, maxx);
            std::uniform_real_distribution<Scalar> dy(miny, maxy);
            std::uniform_real_distribution<Scalar> dang(0, Scalar(2 * Litten_PI));
            std::uniform_real_distribution<Scalar> dr(r, Scalar(2 * r));

            // ---------- grid ----------
            const Scalar cell_size = r / std::sqrt(2);
            const int grid_w = int((maxx - minx) / cell_size) + 1;
            const int grid_h = int((maxy - miny) / cell_size) + 1;

            std::vector<int> grid(grid_w * grid_h, -1);

            auto grid_index = [&](const Vector2 &p)
            {
                int gx = int((p.x() - minx) / cell_size);
                int gy = int((p.y() - miny) / cell_size);
                return std::make_pair(gx, gy);
            };

            auto is_far_enough = [&](const Vector2 &p,
                                     const std::vector<Vector2> &samples)
            {
                auto [gx, gy] = grid_index(p);
                for (int j = -2; j <= 2; ++j)
                {
                    for (int i = -2; i <= 2; ++i)
                    {
                        int nx = gx + i;
                        int ny = gy + j;
                        if (nx < 0 || ny < 0 || nx >= grid_w || ny >= grid_h)
                            continue;
                        int idx = grid[ny * grid_w + nx];
                        if (idx >= 0)
                        {
                            if ((samples[idx] - p).norm() < r)
                                return false;
                        }
                    }
                }
                return true;
            };

            // ---------- init ----------
            std::vector<Vector2> samples;
            std::vector<int> active;

            // pick first point
            for (int it = 0; it < 1000; ++it)
            {
                Vector2 p(dx(rng), dy(rng));
                if (point_in_poly(poly, p))
                {
                    samples.push_back(p);
                    active.push_back(0);
                    auto [gx, gy] = grid_index(p);
                    grid[gy * grid_w + gx] = 0;
                    break;
                }
            }

            // ---------- Bridson loop ----------
            while (!active.empty())
            {
                int idx = active.back();
                active.pop_back();

                const Vector2 &base = samples[idx];
                bool found = false;

                for (int i = 0; i < k; ++i)
                {
                    Scalar ang = dang(rng);
                    Scalar rad = dr(rng);
                    Vector2 p = base + Vector2(std::cos(ang), std::sin(ang)) * rad;

                    if (p.x() < minx || p.x() > maxx ||
                        p.y() < miny || p.y() > maxy)
                        continue;

                    if (!point_in_poly(poly, p))
                        continue;

                    if (!is_far_enough(p, samples))
                        continue;

                    int new_idx = (int)samples.size();
                    samples.push_back(p);
                    active.push_back(new_idx);

                    auto [gx, gy] = grid_index(p);
                    grid[gy * grid_w + gx] = new_idx;

                    found = true;
                }

                if (found)
                    active.push_back(idx);
            }

            return samples;
        }

        // 计算三角形面积
        static Scalar triangle_area(const Vector2 &a, const Vector2 &b, const Vector2 &c)
        {
            return std::abs(((b - a).x() * (c - a).y() - (c - a).x() * (b - a).y()) / Scalar(2));
        }

        // 三角形面积累积和，用于权重采样
        static vector<Scalar> tri_area_cumsum(const vector<Tri> &tris, const vector<Vector2> &verts)
        {
            vector<Scalar> areas(tris.size());
            for (size_t i = 0; i < tris.size(); ++i)
            {
                const auto &t = tris[i];
                areas[i] = std::abs(((verts[t[1]] - verts[t[0]]).x() * (verts[t[2]] - verts[t[0]]).y() - (verts[t[2]] - verts[t[0]]).x() * (verts[t[1]] - verts[t[0]]).y()) / Scalar(2));
            }
            // 累加求和
            std::partial_sum(areas.begin(), areas.end(), areas.begin());
            return areas;
        }

        // 三角形内均匀采样（用重心坐标）
        static Vector2 sample_point_in_triangle(const Vector2 &a, const Vector2 &b, const Vector2 &c, Scalar r0, Scalar r1)
        {
            Scalar sqrt_r0 = std::sqrt(r0);
            return (Scalar(1) - sqrt_r0) * a + (sqrt_r0 * (Scalar(1) - r1)) * b + (sqrt_r0 * r1) * c;
        }

        // 从面积累计分布中采样三角形索引
        static size_t sample_triangle_index(const vector<Scalar> &cumsum_area, Scalar r)
        {
            Scalar total_area = cumsum_area.back();
            Scalar target = r * total_area;
            auto it = std::lower_bound(cumsum_area.begin(), cumsum_area.end(), target);
            return std::distance(cumsum_area.begin(), it);
        }

        // === nanoflann KDTree 适配器，基于 std::vector<Vector2> ===
        struct PointCloudAdaptor
        {
            const vector<Vector2> &pts;
            PointCloudAdaptor(const vector<Vector2> &points) : pts(points) {}

            inline size_t kdtree_get_point_count() const { return pts.size(); }
            inline Scalar kdtree_get_pt(const size_t idx, const size_t dim) const
            {
                return pts[idx][dim];
            }
            template <class BBOX>
            bool kdtree_get_bbox(BBOX &) const { return false; }
        };

        using KDTreeType = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<Scalar, PointCloudAdaptor>,
            PointCloudAdaptor,
            2 /* dim */
            >;

        // 核心函数：泊松盘采样
        static std::vector<Vector2> poisson_disk_sample(
            const std::vector<Vector2> &vertices,
            const std::vector<Tri> &triangles,
            Scalar radius,
            size_t num_iterations,
            std::mt19937 &rng)
        {
            std::vector<Vector2> samples;

            if (triangles.empty() || vertices.empty())
                return samples;

            // 计算面积累计
            std::vector<Scalar> tri_cumsum_area = tri_area_cumsum(triangles, vertices);

            std::uniform_real_distribution<Scalar> uniform_dist(0.0, 1.0);

            KDTreeType *kd_tree = nullptr;

            for (size_t iter = 0; iter < num_iterations; ++iter)
            {
                // 采样三角形
                Scalar r_tri = uniform_dist(rng);
                size_t tri_idx = sample_triangle_index(tri_cumsum_area, r_tri);
                const auto &tri = triangles[tri_idx];

                // 在三角形内均匀采样点
                Scalar r0 = uniform_dist(rng);
                Scalar r1 = uniform_dist(rng);
                Vector2 p = sample_point_in_triangle(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]], r0, r1);

                // 第一个点，直接加入并建树
                if (samples.empty())
                {
                    samples.push_back(p);
                    // 用新的临时 PointCloudAdaptor 构造 KDTree
                    PointCloudAdaptor pc(samples);
                    kd_tree = new KDTreeType(2, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                    kd_tree->buildIndex();
                    continue;
                }

                // 查询半径内是否已有点
                const size_t max_results = 1;
                std::vector<nanoflann::ResultItem<uint32_t, Scalar>> ret_matches(max_results);
                nanoflann::SearchParameters params; // nanoflann 1.8 新版参数

                size_t n_matches = kd_tree->radiusSearch(p.data(), radius * radius, ret_matches, params);

                if (n_matches == 0)
                {
                    // 没有点距离太近，可以加入
                    samples.push_back(p);

                    // 重新构建 KDTree，先释放旧树
                    delete kd_tree;

                    PointCloudAdaptor pc(samples);
                    kd_tree = new KDTreeType(2, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                    kd_tree->buildIndex();
                }
                // 否则丢弃该点，继续下一次采样
            }

            if (kd_tree)
                delete kd_tree;

            return samples;
        }

        // 将Eigen点转成 earcut 所需格式
        static std::vector<std::vector<std::array<Scalar, 2>>> convert_to_earcut(const std::vector<Vector2> &polygon)
        {
            using Point = std::array<Scalar, 2>;
            using Ring = std::vector<Point>;

            Ring ring;
            ring.reserve(polygon.size());

            for (const auto &p : polygon)
            {
                ring.push_back({Scalar(p.x()), Scalar(p.y())});
            }

            // 保证CCW
            if (signed_polygon_area(polygon) < Scalar(0))
                std::reverse(ring.begin(), ring.end());

            std::vector<Ring> out;
            out.push_back(ring);
            return out;
        }

        // 额外的三角剖分并调用泊松采样接口
        static std::vector<Vector2> poisson_disk_sample_from_polygon(
            const std::vector<Vector2> &polygon,
            Scalar radius,
            size_t num_iterations,
            unsigned seed = 0)
        {
            if (polygon.size() < 3)
                return {};

            // 转换格式给 earcut
            auto polygon_earcut = convert_to_earcut(polygon);

            // earcut 返回的是索引序列，3个一组是三角形
            std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(polygon_earcut);

            // 转换为 Tri 类型的三角形索引数组
            std::vector<Tri> triangles;
            for (size_t i = 0; i + 2 < indices.size(); i += 3)
            {
                triangles.push_back({indices[i], indices[i + 1], indices[i + 2]});
            }

            std::mt19937 rng(seed);
            // 调用之前写的 poisson_disk_sample 函数（需修改成非static 或用Math2::poisson_disk_sample调用）
            return poisson_disk_sample(polygon, triangles, radius, num_iterations, rng);
        }

        static std::vector<Tri> triangulate_poly(const std::vector<Vector2> &polygon2d)
        {
            if (polygon2d.size() < 3)
                return {};

            // 转换格式给 earcut
            auto polygon_earcut = convert_to_earcut(polygon2d);

            // earcut 返回的是索引序列，3个一组是三角形
            std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(polygon_earcut);

            // 转换为 Tri 类型的三角形索引数组
            std::vector<Tri> triangles;
            for (size_t i = 0; i + 2 < indices.size(); i += 3)
            {
                triangles.push_back({indices[i], indices[i + 1], indices[i + 2]});
            }
            return triangles;
        }

        static bool compute_plane(const std::vector<Vector3> &pts,Vector3 &origin,Vector3 &normal,Scalar eps)
        {
            if (pts.size() < 3)
                return false;

            origin = pts[0];

            // 找不共线的三点
            Vector3 n = Vector3::Zero();
            for (size_t i = 1; i + 1 < pts.size(); ++i)
            {
                Vector3 a = pts[i] - origin;
                Vector3 b = pts[i + 1] - origin;
                n = a.cross(b);
                if (n.norm() > eps)
                    break;
            }

            if (n.norm() <= eps)
                return false; // 全共线

            normal = n.normalized();

            // 检查所有点到平面的距离
            for (auto &p : pts)
            {
                Scalar d = (p - origin).dot(normal);
                if (std::fabs(d) > eps)
                    return false; // 不共面
            }

            return true;
        }

        static void make_plane_basis(const Vector3 &n,Vector3 &u,Vector3 &v)
        {
            Vector3 tmp = (std::fabs(n.x()) < Scalar(0.9))
                              ? Vector3(1, 0, 0)
                              : Vector3(0, 1, 0);

            u = n.cross(tmp).normalized();
            v = u.cross(n);
        }

        static std::vector<Vector2> project_to_2d(
            const std::vector<Vector3> &pts,
            const Vector3 &origin,
            const Vector3 &u,
            const Vector3 &v)
        {
            std::vector<Vector2> out;
            out.reserve(pts.size());

            for (const auto &p : pts)
            {
                Vector3 d = p - origin;
                Vector2 p2d;
                p2d << d.dot(u), d.dot(v);
                out.push_back(p2d);
            }

            return out;
        }
    };

}