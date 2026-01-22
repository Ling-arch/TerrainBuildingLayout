#pragma once
#include <Eigen/Core>
#include "polyloop.h"
#include <raylib.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Straight_skeleton_2.h>
#include <CGAL/create_offset_polygons_2.h>
#include <CGAL/Polygon_offset_builder_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <list>
#include "util.h"

namespace geo
{

    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point_2 = Kernel::Point_2;
    using Segment_2 = Kernel::Segment_2;
    using Polygon2 = CGAL::Polygon_2<Kernel>;
    using PolygonWithHoles = CGAL::Polygon_with_holes_2<Kernel>;
    using Traits = CGAL::Arr_segment_traits_2<Kernel>;
    using Arrangement = CGAL::Arrangement_2<Traits>;
    using Ss = CGAL::Straight_skeleton_2<Kernel>;

    template <typename Scalar>
    using Vector2 = Eigen::Matrix<Scalar, 2, 1>;

    struct Segment
    {
        Eigen::Vector3f p0;
        Eigen::Vector3f p1;
    };

    struct PointKey
    {
        int x, y, z;
        bool operator==(const PointKey &o) const
        {
            return x == o.x && y == o.y && z == o.z;
        }
    };

    PointKey makeKey(const Eigen::Vector3f &p, float eps);

    struct PointKeyHash
    {
        size_t operator()(const PointKey &k) const
        {
            return ((size_t)k.x * 73856093u) ^
                   ((size_t)k.y * 19349663u) ^
                   ((size_t)k.z * 83492791u);
        }
    };

    struct Polyline3
    {
        std::vector<Eigen::Vector3f> points;
        bool closed = false;
    };

    struct Edge
    {
        int segId;
        int end; // 0 = p0, 1 = p1
    };

    struct GraphEdge
    {
        int from;
        int to;
        float cost;
        int dx;
        int dy;
    };

    struct Circle
    {
        Eigen::Vector2f center; // face 质心
        float radius;
    };

    std::vector<Polyline3> buildPolylines(const std::vector<geo::Segment> &segments, float eps = 1e-4f);

    void followChain(
        int startSeg,
        int startEnd,
        const std::vector<geo::Segment> &segments,
        const std::unordered_map<PointKey, std::vector<Edge>, PointKeyHash> &adj,
        std::vector<bool> &used,
        float eps,
        Polyline3 &out);

    struct Vertex
    {
        Eigen::Vector3f position;
        Eigen::Vector3f normal;
    };

    struct MeshData
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices; // triangle list
    };

    class PolygonMesh
    {
    public:
        MeshData mesh;
        PolygonMesh() = default;
        PolygonMesh(const std::vector<Eigen::Vector3f> &points_, float height_)
            : points(points_), height(height_)
        {
            build();
            upload();
        }
        void build();
        void upload();
        void regenerate(float newHeight);
        void draw(Color color, float colorAlpha, bool outline, bool wireframe, float wireframeAlpha);

    private:
        std::vector<Eigen::Vector3f> points;
        float height;
        Model model;
    };

    Mesh buildRaylibMesh(const MeshData &src);

    template <typename Scalar>
    bool pointCompare(const Vector2<Scalar> &a, const Vector2<Scalar> &b)
    {
        if (a.x() != b.x())
            return a.x() < b.x();
        return a.y() < b.y();
    }

    /*
     *cross product of Eigen vector2
     */
    template <typename Scalar>
    Scalar cross(const Vector2<Scalar> &a,
                 const Vector2<Scalar> &b,
                 const Vector2<Scalar> &c)
    {
        return (b - a).x() * (c - a).y() - (b - a).y() * (c - a).x();
    }

    /*
     * if s0(a,b) & s1(c,d) intersect
     */
    template <typename Scalar>
    bool segmentsIntersect(const Vector2<Scalar> &a, const Vector2<Scalar> &b, const Vector2<Scalar> &c, const Vector2<Scalar> &d)
    {
        Scalar c1 = cross(a, b, c);
        Scalar c2 = cross(a, b, d);
        Scalar c3 = cross(c, d, a);
        Scalar c4 = cross(c, d, b);

        return (c1 * c2 < 0) && (c3 * c4 < 0);
    }

    /*
     *polyline2 struct with closed polygon and polyline
     */
    template <typename Scalar>
    struct Polyline2_t
    {
        std::vector<Vector2<Scalar>> points;
        bool isClosed = false;
        Polyline2_t() = default;
        Polyline2_t(const std::vector<Vector2<Scalar>> &pts, bool isClosed)
            : points(pts), isClosed(isClosed)
        {
            if (isClosed && (pts.front() - pts.back()).squaredNorm() > Scalar(1e-6))
            {
                points.push_back(pts.front());
            }
        }
        Polyline2_t(const std::vector<Vector2<Scalar>> &pts)
            : points(pts)
        {
            if (isClosed && (pts.front() - pts.back()).squaredNorm() > Scalar(1e-6))
            {
                points.push_back(pts.front());
            }
        }
        Vector2<Scalar> getTangentAt(int idx) const;
        Vector2<Scalar> getNormalAt(int idx) const;
        Eigen::AlignedBox<Scalar, 2> getAABB2() const;
        bool isSelfIntersecting() const;
    };

    /*
     *get tangent vector at point with point index
     */
    template <typename Scalar>
    Vector2<Scalar> Polyline2_t<Scalar>::getTangentAt(int idx) const
    {
        const int n = static_cast<int>(points.size());
        if (n < 2)
            return Vector2<Scalar>(1, 0);

        int i0, i1;

        if (isClosed)
        {
            i0 = (idx - 1 + n) % n;
            i1 = (idx + 1) % n;
            if (idx == n - 1 || idx == 0)
            {
                i0 = n - 2;
                i1 = 1;
            }
        }
        else
        {
            if (idx == 0)
            {
                i0 = 0;
                i1 = 1;
            }
            else if (idx == n - 1)
            {
                i0 = n - 2;
                i1 = n - 1;
            }
            else
            {
                i0 = idx - 1;
                i1 = idx + 1;
            }
        }

        Vector2<Scalar> t = points[i1] - points[i0];
        Scalar len = t.norm();
        if (len < Scalar(1e-6))
            return Vector2<Scalar>(1, 0);

        return t / len;
    }

    /*
     *get normal vector at point with point index
     */
    template <typename Scalar>
    Vector2<Scalar> Polyline2_t<Scalar>::getNormalAt(int idx) const
    {
        Vector2<Scalar> tangent = getTangentAt(idx);
        return Vector2<Scalar>(-tangent.y(), tangent.x());
    }

    template <typename Scalar>
    bool Polyline2_t<Scalar>::isSelfIntersecting() const
    {

        int n = (int)points.size();

        // 闭合多边形至少需要 4 个点（含重复起点）
        if (isClosed)
        {
            if (n < 4)
                return false;
        }
        else
        {
            if (n < 4)
                return false; // 折线至少 3 段才可能自交
        }

        int edgeCount = n - 1;

        for (int i = 0; i < edgeCount; ++i)
        {
            int i0 = i;
            int i1 = i + 1; // 对闭合 polygon，最后一条边是 (n-2 -> n-1)，n-1 == p0

            for (int j = i + 1; j < edgeCount; ++j)
            {
                // 跳过相邻边
                if (std::abs(i - j) <= 1)
                    continue;

                // 闭合 polygon：第一条边 和 最后一条边 也是相邻的
                if (isClosed && i == 0 && j == edgeCount - 1)
                    continue;

                int j0 = j;
                int j1 = j + 1;

                if (segmentsIntersect(points[i0], points[i1], points[j0], points[j1]))
                {
                    return true;
                }
            }
        }
        return false;
    }

    /*
     *rortate vector clock wise 90 degree
     */
    template <typename Scalar>
    inline Vector2<Scalar> rotate90CW(const Vector2<Scalar> &v)
    {
        return Vector2<Scalar>(v.y(), -v.x());
    }

    template <typename Scalar>
    inline Vector2<Scalar> rotate90CCW(const Vector2<Scalar> &v)
    {
        return Vector2<Scalar>(-v.y(), v.x());
    }

    /*
     *get polyline2 aabb
     */
    template <typename Scalar>
    inline Eigen::AlignedBox<Scalar, 2> Polyline2_t<Scalar>::getAABB2() const
    {
        Eigen::AlignedBox<Scalar, 2> aabb;
        aabb.setEmpty();
        if (points.empty())
            return aabb;

        for (const auto &p : points)
            aabb.extend(p);
        return aabb;
    }

    /*
     *get points aabb
     */
    template <typename Scalar>
    inline Eigen::AlignedBox<Scalar, 2> getAABB2(const std::vector<Vector2<Scalar>> &points)
    {
        Eigen::AlignedBox<Scalar, 2> aabb;
        aabb.setEmpty();
        if (points.empty())
            return aabb;

        for (const auto &p : points)
            aabb.extend(p);
        return aabb;
    }

    template <typename Scalar>
    inline Vector2<Scalar> getPolygonCentroid(const std::vector<Vector2<Scalar>> &points)
    {
        if (points.empty())
        {
            return Vector2<Scalar>::Zero();
        }
        Vector2<Scalar> centroid = Vector2<Scalar>::Zero();
        for (const auto &pt : points)
            centroid += pt;

        return centroid / static_cast<Scalar>(points.size());
    }

    // 获取凸包的函数
    template <typename Scalar>
    std::vector<Vector2<Scalar>> getConvexHull(const std::vector<Vector2<Scalar>> &points)
    {
        int n = (int)points.size();
        if (n <= 3)
            return points; // 0~3点本身就是凸包

        // 1. 按 x, y 排序
        std::vector<Vector2<Scalar>> pts = points;
        std::sort(pts.begin(), pts.end(), pointCompare<Scalar>);

        std::vector<Vector2<Scalar>> hull(2 * n);
        int k = 0;

        // 2. 构造下凸包
        for (int i = 0; i < n; ++i)
        {
            while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0)
            {
                k--;
            }
            hull[k++] = pts[i];
        }

        // 3. 构造上凸包
        for (int i = n - 2, t = k + 1; i >= 0; --i)
        {
            while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0)
            {
                k--;
            }
            hull[k++] = pts[i];
        }

        hull.resize(k - 1); // 去掉重复的最后一点

        hull.push_back(hull.front());
        return hull;
    }

    template <typename Scalar>
    struct OBB2
    {
        std::vector<Vector2<Scalar>> points;
        Vector2<Scalar> center;
        Vector2<Scalar> axis0;    // 主轴（单位向量）
        Vector2<Scalar> axis1;    // 副轴（axis0 ⟂）
        Vector2<Scalar> halfSize; // 在 axis0 / axis1 上的半长
        Polyline2_t<Scalar> poly;

        OBB2() = default;
        OBB2(const std::vector<Vector2<Scalar>> &points_)
            : points(points_)
        {
            // computeObb2();
            computeObbMinArea();
        }
        void computeObb2();
        void computeObbMinArea();
    };

    /*
     * compute Obb with Eigen PCA
     */
    template <typename Scalar>
    void OBB2<Scalar>::computeObb2()
    {
        using MatrixX2 = Eigen::Matrix<Scalar, Eigen::Dynamic, 2>;
        int n = (int)points.size();
        if (n == 0)
            return;

        // 1. 计算中心（均值）
        Vector2<Scalar> mean = Vector2<Scalar>::Zero();
        for (const auto &p : points)
            mean += p;
        mean /= Scalar(n);
        center = mean;

        // 2. 构建协方差矩阵
        MatrixX2 centered(n, 2);
        for (int i = 0; i < n; ++i)
            centered.row(i) = points[i] - mean;

        Eigen::Matrix<Scalar, 2, 2> cov = (centered.transpose() * centered) / Scalar(n);

        // 3. 对协方差矩阵做特征值分解（PCA）
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> eig(cov);
        if (eig.info() != Eigen::Success)
            return;

        // 4. 特征向量，按特征值大小排序，最大特征值对应主轴
        Eigen::Matrix<Scalar, 2, 2> eigVecs = eig.eigenvectors(); // 列是特征向量，默认升序
        // 由于是升序，最大特征值对应第二列
        axis0 = eigVecs.col(1).normalized();
        axis1 = eigVecs.col(0).normalized(); // 正交方向

        // 5. 投影所有点到主轴 axis0 和 axis1 上，找最大最小投影值
        Scalar min0 = std::numeric_limits<Scalar>::max();
        Scalar max0 = -std::numeric_limits<Scalar>::max();
        Scalar min1 = std::numeric_limits<Scalar>::max();
        Scalar max1 = -std::numeric_limits<Scalar>::max();

        for (const auto &p : points)
        {
            Vector2<Scalar> d = p - mean;
            Scalar proj0 = axis0.dot(d);
            Scalar proj1 = axis1.dot(d);

            min0 = std::min(min0, proj0);
            max0 = std::max(max0, proj0);
            min1 = std::min(min1, proj1);
            max1 = std::max(max1, proj1);
        }

        // 6. 计算半尺寸
        // Vector2<Scalar> halfSize;
        halfSize.x() = (max0 - min0) / Scalar(2);
        halfSize.y() = (max1 - min1) / Scalar(2);

        // 7. 计算中心在主轴方向的坐标
        Scalar center0 = (max0 + min0) / Scalar(2);
        Scalar center1 = (max1 + min1) / Scalar(2);

        center = mean + axis0 * center0 + axis1 * center1;

        // 8. 构造多边形顶点，顺时针或逆时针，且闭合
        std::vector<Vector2<Scalar>> obbPoints(5);
        obbPoints[0] = center + axis0 * halfSize.x() + axis1 * halfSize.y();
        obbPoints[1] = center - axis0 * halfSize.x() + axis1 * halfSize.y();
        obbPoints[2] = center - axis0 * halfSize.x() - axis1 * halfSize.y();
        obbPoints[3] = center + axis0 * halfSize.x() - axis1 * halfSize.y();
        obbPoints[4] = obbPoints[0]; // 闭合多边形最后一个点等于第一个点

        poly = Polyline2_t<Scalar>(obbPoints, true);
    }

    /*
     * compute Obb with rotation clipper
     */
    template <typename Scalar>
    void OBB2<Scalar>::computeObbMinArea()
    {
        // 0. 输入检查
        if (points.size() < 2)
            return;

        // 1. 用你已有的函数构造凸包
        std::vector<Vector2<Scalar>> hull = getConvexHull(points);
        const int m = (int)hull.size();
        if (m < 2)
            return;

        Scalar bestArea = std::numeric_limits<Scalar>::max();

        Vector2<Scalar> bestCenter;
        Vector2<Scalar> bestAxis0;
        Vector2<Scalar> bestAxis1;
        Vector2<Scalar> bestHalfSize;

        // 2. rotating calipers：遍历每条 hull 边
        for (int i = 0; i < m; ++i)
        {
            const Vector2<Scalar> &p0 = hull[i];
            const Vector2<Scalar> &p1 = hull[(i + 1) % m];

            Vector2<Scalar> axis0 = p1 - p0;
            Scalar len = axis0.norm();
            if (len < Scalar(1e-8))
                continue;

            axis0 /= len;
            Vector2<Scalar> axis1(-axis0.y(), axis0.x());

            Scalar min0 = std::numeric_limits<Scalar>::max();
            Scalar max0 = -min0;
            Scalar min1 = min0;
            Scalar max1 = -min0;

            for (const auto &p : hull)
            {
                Scalar d0 = axis0.dot(p);
                Scalar d1 = axis1.dot(p);

                min0 = std::min(min0, d0);
                max0 = std::max(max0, d0);
                min1 = std::min(min1, d1);
                max1 = std::max(max1, d1);
            }

            Scalar area = (max0 - min0) * (max1 - min1);
            if (area < bestArea)
            {
                bestArea = area;

                bestAxis0 = axis0;
                bestAxis1 = axis1;

                bestHalfSize.x() = (max0 - min0) * Scalar(0.5);
                bestHalfSize.y() = (max1 - min1) * Scalar(0.5);

                Scalar c0 = (max0 + min0) * Scalar(0.5);
                Scalar c1 = (max1 + min1) * Scalar(0.5);
                bestCenter = axis0 * c0 + axis1 * c1;
            }
        }

        // 3. 写回 OBB 成员变量
        center = bestCenter;
        axis0 = bestAxis0;
        axis1 = bestAxis1;
        halfSize = bestHalfSize;

        // 4. 构造可视化 poly（与你之前 PCA 版本完全一致）
        std::vector<Vector2<Scalar>> obbPoints(5);
        obbPoints[0] = center + axis0 * halfSize.x() + axis1 * halfSize.y();
        obbPoints[1] = center - axis0 * halfSize.x() + axis1 * halfSize.y();
        obbPoints[2] = center - axis0 * halfSize.x() - axis1 * halfSize.y();
        obbPoints[3] = center + axis0 * halfSize.x() - axis1 * halfSize.y();
        obbPoints[4] = obbPoints[0]; // 闭合

        poly = Polyline2_t<Scalar>(obbPoints, true);
    }

    template <typename Scalar>
    inline Polygon2 toCgalPolygon(const Polyline2_t<Scalar> &poly)
    {
        Polygon2 p;
        int n;
        if ((poly.points.front() - poly.points.back()).squaredNorm() < Scalar(1e-6))
            n = (int)poly.points.size() - 1;
        else
            n = (int)poly.points.size();

        for (int i = 0; i < n; ++i) // 去掉闭合重复点
        {
            p.push_back(Point_2(typename Kernel::FT(poly.points[i].x()), typename Kernel::FT(poly.points[i].y())));
        }

        if (p.orientation() != CGAL::COUNTERCLOCKWISE)
            p.reverse_orientation();

        return p;
    }

    /*
     * convert Polyline to CGAL Polygon
     */
    template <typename Scalar>
    inline Polyline2_t<Scalar> cgalPolygonToPolyline(const Polygon2 &poly)
    {
        Polyline2_t<Scalar> out;

        if (poly.is_empty())
            return out;

        std::vector<Vector2<Scalar>> pts;
        pts.reserve(poly.size() + 1);

        for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it)
        {
            Scalar x = static_cast<Scalar>(CGAL::to_double(it->x()));
            Scalar y = static_cast<Scalar>(CGAL::to_double(it->y()));

            pts.emplace_back(x, y);
        }

        // 闭合
        if (!pts.empty())
            pts.push_back(pts.front());

        out.points = std::move(pts);
        out.isClosed = true;

        return out;
    }

    template <typename Scalar>
    Polyline2_t<Scalar> unionPolygon(const Polyline2_t<Scalar> &polyA, const Polyline2_t<Scalar> &polyB)
    {
        std::vector<Polyline2_t<Scalar>> polys;
        Polygon2 A = toCgalPolygon(polyA);
        Polygon2 B = toCgalPolygon(polyB);

        PolygonWithHoles result;

        if (CGAL::join(A, B, result))
            return cgalPolygonToPolyline<Scalar>(result.outer_boundary());
        else
            return polyA;
    }

    template <typename Scalar>
    std::vector<Polyline2_t<Scalar>> subPolygon(const Polyline2_t<Scalar> &polyA, const Polyline2_t<Scalar> &polyB)
    {
        std::vector<Polyline2_t<Scalar>> polys;
        Polygon2 A = toCgalPolygon(polyA);
        Polygon2 B = toCgalPolygon(polyB);

        std::list<PolygonWithHoles> result;
        CGAL::difference(A, B, std::back_inserter(result));

        if (result.empty())
            return {polyA};

        for (const PolygonWithHoles &p : result)
            polys.push_back(cgalPolygonToPolyline<Scalar>(p.outer_boundary()));
        return polys;
    }

    template <typename Scalar>
    std::vector<Polyline2_t<Scalar>> intersectPolygon(const Polyline2_t<Scalar> &polyA, const Polyline2_t<Scalar> &polyB)
    {

        std::vector<Polyline2_t<Scalar>> polys;
        Polygon2 A = toCgalPolygon(polyA);
        Polygon2 B = toCgalPolygon(polyB);

        std::list<PolygonWithHoles> result;
        CGAL::intersection(A, B, std::back_inserter(result));

        if (result.empty())
            return {{polyA}};

        for (const PolygonWithHoles &p : result)
            polys.push_back(cgalPolygonToPolyline<Scalar>(p.outer_boundary()));
        return polys;
    }


    /*
    *offset polygon,
    */
    template <typename Scalar>
    std::vector<Polyline2_t<Scalar>> offsetPolygon(const Polyline2_t<Scalar> &poly, Scalar dist)
    {
        std::vector<Polyline2_t<Scalar>> out;
        using Ss = CGAL::Straight_skeleton_2<Kernel>;
        using SsPtr = std::shared_ptr<Ss>;
        using OffsetPolygonPtr = std::shared_ptr<Polygon2>;
        Polygon2 cgalPoly = toCgalPolygon(poly);

        // 先生成 Straight Skeleton
        SsPtr ss = CGAL::create_interior_straight_skeleton_2(cgalPoly);

        if (!ss)
            return out;

        //  再从 skeleton 生成 offset polygon
        auto offset_polys = CGAL::create_offset_polygons_2<Polygon2>(-dist, *ss);

        for (const auto &op : offset_polys)
            out.push_back(cgalPolygonToPolyline<Scalar>(*op));

        return out;
    }

    template <typename Scalar>
    void insertPolygonEdges(const Polyline2_t<Scalar> &poly, Arrangement &arr)
    {
        int n = (int)poly.points.size();
        if (n < 3)
            return;
        for (int i = 0; i < n - 1; ++i)
        {
            const auto &p0 = poly.points[i];
            const auto &p1 = poly.points[(i + 1) % n];

            if ((p0 - p1).squaredNorm() < (Scalar)1e-8)
                continue;

            CGAL::insert(arr, Segment_2(Point_2(typename Kernel::FT(p0.x()), typename Kernel::FT(p0.y())),
                                        Point_2(typename Kernel::FT(p1.x()), typename Kernel::FT(p1.y()))));
        }
    }

    template <typename Scalar>
    void insertPolyline(const Polyline2_t<Scalar> &pl, Arrangement &arr)
    {
        int n = (int)pl.points.size();
        for (int i = 0; i + 1 < n; ++i)
        {
            const auto &p0 = pl.points[i];
            const auto &p1 = pl.points[i + 1];

            if ((p0 - p1).squaredNorm() < 1e-12f)
                continue;

            CGAL::insert(arr, Segment_2(Point_2(typename Kernel::FT(p0.x()), typename Kernel::FT(p0.y())),
                                        Point_2(typename Kernel::FT(p1.x()), typename Kernel::FT(p1.y()))));
        }
    }

    template <typename Scalar>
    std::vector<Polyline2_t<Scalar>> extractFaces(const Arrangement &arr, const Polyline2_t<Scalar> &originalPoly)
    {
        std::vector<Polyline2_t<Scalar>> result;
        for (auto fit = arr.faces_begin(); fit != arr.faces_end(); ++fit)
        {
            if (fit->is_unbounded())
                continue;
            auto ccb = fit->outer_ccb();
            if (ccb == nullptr)
                continue;
            std::vector<Vector2<Scalar>> ring;
            auto curr = ccb;
            do
            {
                auto src = curr->source()->point();
                ring.emplace_back(static_cast<Scalar>(CGAL::to_double(src.x())), static_cast<Scalar>(CGAL::to_double(src.y())));
                ++curr;
            } while (curr != ccb);
            Vector2<Scalar> center = util::Math2<Scalar>::getPointStrictInSidePolygon(ring);
            if (util::Math2<Scalar>::point_in_poly(originalPoly.points, center))
            {
                Polyline2_t<Scalar> poly;
                poly.points = ring;
                poly.isClosed = true;
                poly.points.push_back(poly.points.front());
                result.push_back(poly);
            }
        }
        return result;
    }

    template <typename Scalar>
    std::vector<Polyline2_t<Scalar>> splitPolygonByPolylines(const Polyline2_t<Scalar> &polygon, const std::vector<Polyline2_t<Scalar>> &cutters)
    {
        Arrangement arr;
        insertPolygonEdges(polygon, arr);
        for (const auto &pl : cutters)
            insertPolyline(pl, arr);
        return extractFaces(arr, polygon);
    }
}