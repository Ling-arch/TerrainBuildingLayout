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
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/Surface_mesh.h>

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
    using Vb = CGAL::Triangulation_vertex_base_2<Kernel>;
    using Fb = CGAL::Constrained_triangulation_face_base_2<Kernel>;
    using TDS = CGAL::Triangulation_data_structure_2<Vb, Fb>;
    using CDT = CGAL::Constrained_Delaunay_triangulation_2<Kernel, TDS>;
    using AT = CGAL::Delaunay_triangulation_adaptation_traits_2<CDT>;
    using AP = CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<CDT>;
    using VD = CGAL::Voronoi_diagram_2<CDT, AT, AP>;
    using Point_3 = Kernel::Point_3;
    using SurfaceMesh = CGAL::Surface_mesh<Point_3>;

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
        Eigen::Vector3f normal = {0.f, 0.f, 1.f};
        Vertex() = default;
        Vertex(const Eigen::Vector3f &pos) : position(pos) {}
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

    inline void computeVertexNormals(MeshData &mesh)
    {
        // 1. 清零法线
        for (auto &v : mesh.vertices)
        {
            v.normal.setZero();
        }

        // 2. 遍历三角形，累加面法线
        const size_t triCount = mesh.indices.size() / 3;

        for (size_t t = 0; t < triCount; ++t)
        {
            uint32_t i0 = mesh.indices[t * 3 + 0];
            uint32_t i1 = mesh.indices[t * 3 + 1];
            uint32_t i2 = mesh.indices[t * 3 + 2];

            const Eigen::Vector3f &p0 = mesh.vertices[i0].position;
            const Eigen::Vector3f &p1 = mesh.vertices[i1].position;
            const Eigen::Vector3f &p2 = mesh.vertices[i2].position;

            Eigen::Vector3f e1 = p1 - p0;
            Eigen::Vector3f e2 = p2 - p0;

            Eigen::Vector3f faceNormal = e1.cross(e2);

            float len2 = faceNormal.squaredNorm();
            if (len2 < 1e-12f)
                continue; // 退化三角形

            // 面法线按面积权重累加
            mesh.vertices[i0].normal += faceNormal;
            mesh.vertices[i1].normal += faceNormal;
            mesh.vertices[i2].normal += faceNormal;
        }

        // 3. 归一化顶点法线
        for (auto &v : mesh.vertices)
        {
            float len = v.normal.norm();
            if (len > 1e-6f)
            {
                v.normal /= len;
            }
            else
            {
                // 兜底法线（例如完全平面或孤立点）
                v.normal = Eigen::Vector3f(0.f, 0.f, 1.f);
            }
        }
    }
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

    template <typename Scalar>
    bool trianglesDoIntersect2D(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T1,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T2)
    {
        using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

        Vec2 t1[3], t2[3];

        for (int i = 0; i < 3; ++i)
        {
            t1[i] = T1.row(i).template head<2>().transpose();
            t2[i] = T2.row(i).template head<2>().transpose();
        }

        // edge-edge intersection
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (segmentsIntersect<Scalar>(
                        t1[i],
                        t1[(i + 1) % 3],
                        t2[j],
                        t2[(j + 1) % 3]))
                    return true;
            }
        }

        // T2 inside T1
        bool allInside = true;
        for (int i = 0; i < 3; ++i)
        {
            if (!util::Math2<Scalar>::point_in_poly(
                    {t1[0], t1[1], t1[2]}, t2[i]))
            {
                allInside = false;
                break;
            }
        }

        if (allInside)
            return true;

        // T1 inside T2
        allInside = true;
        for (int i = 0; i < 3; ++i)
        {
            if (!util::Math2<Scalar>::point_in_poly(
                    {t2[0], t2[1], t2[2]}, t1[i]))
            {
                allInside = false;
                break;
            }
        }

        return allInside;
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
    Polyline2_t<Scalar> rebuildPolyline(const Polyline2_t<Scalar> &polyline, Scalar threshold)
    {
        Polyline2_t<Scalar> result;
        result.isClosed = polyline.isClosed;

        const auto &pts = polyline.points;
        if (pts.size() < 2 || threshold <= Scalar(0))
        {
            result.points = pts;
            return result;
        }

        const int segCount = static_cast<int>(pts.size()) - 1;

        for (int i = 0; i < segCount; ++i)
        {
            const Vector2<Scalar> &p0 = pts[i];
            const Vector2<Scalar> &p1 = pts[(i + 1) % pts.size()];

            // 先压入起点
            result.points.push_back(p0);

            Scalar dist = (p1 - p0).norm();
            int n = static_cast<int>(std::floor(dist / threshold));

            if (n > 1)
            {
                // 均匀插值
                for (int k = 1; k < n; ++k)
                {
                    Scalar t = Scalar(k) / Scalar(n);
                    Vector2<Scalar> p = (Scalar(1) - t) * p0 + t * p1;
                    result.points.push_back(p);
                }
            }
        }

        // 非闭合 polyline，补最后一个点
        if (!polyline.isClosed)
        {
            result.points.push_back(pts.back());
        }

        return result;
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
     *offset polygon,negative distance for inward, positive for outward, return multiple offset polygons because of possible self-intersection
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

    template <typename Scalar>
    struct VoronoiResult2
    {
        CDT cdt;
        std::vector<std::pair<Vector2<Scalar>, Vector2<Scalar>>> voronoi_edges;
        std::vector<Polyline2_t<Scalar>> voronoi_cells;
        VoronoiResult2() = default;
    };

    /*
     *convert eigen vector to cgal point
     */
    template <typename Scalar>
    inline Point_2 to_cgal_point(const Vector2<Scalar> &p)
    {
        return Point_2(typename Kernel::FT(p.x()), typename Kernel::FT(p.y()));
    }

    template <typename Scalar>
    inline Point_3 to_cgal_point(const Vector2<Scalar> &p, Scalar z)
    {
        return Point_3(typename Kernel::FT(p.x()), typename Kernel::FT(p.y()), typename Kernel::FT(z));
    }

    template <typename Scalar>
    Polyline2_t<Scalar> extract_voronoi_cell(const VD::Face_handle &face)
    {
        Polyline2_t<Scalar> poly;
        poly.isClosed = true;

        VD::Ccb_halfedge_circulator circ = face->ccb();
        VD::Ccb_halfedge_circulator curr = circ;

        do
        {
            if (curr->has_source())
            {
                const auto &p = curr->source()->point();
                poly.points.emplace_back(
                    static_cast<Scalar>(p.x()),
                    static_cast<Scalar>(p.y()));
            }
            ++curr;
        } while (curr != circ);

        // 闭合
        if (!poly.points.empty() &&
            (poly.points.front() - poly.points.back()).squaredNorm() > Scalar(1e-6))
        {
            poly.points.push_back(poly.points.front());
        }

        return poly;
    }

    template <typename Scalar>
    VoronoiResult2<Scalar> build_voronoi_from_pslg(const Polyline2_t<Scalar> &boundary, const std::vector<Polyline2_t<Scalar>> &constraints)
    {
        VoronoiResult2<Scalar> result;
        CDT &cdt = result.cdt;
        int boundary_pt_num = 0;
        if ((boundary.points.back() - boundary.points.front()).squaredNorm() < (Scalar)1e-6)
            boundary_pt_num = boundary.points.size() - 1;
        else
            boundary_pt_num = boundary.points.size();
        for (int i = 0; i < boundary_pt_num; i++)
        {
            cdt.insert_constraint(to_cgal_point(boundary.points[i]), to_cgal_point(boundary.points[(i + 1) % boundary_pt_num]));
        }

        if (constraints.size() > 0)
        {
            for (const auto &pl : constraints)
            {
                const auto &pts = pl.points;
                const int n = static_cast<int>(pts.size());

                for (int i = 0; i + 1 < n; ++i)
                {
                    cdt.insert_constraint(to_cgal_point(pts[i]), to_cgal_point(pts[i + 1]));
                }
            }
        }

        VD vd(cdt);
        for (auto fit = vd.faces_begin(); fit != vd.faces_end(); ++fit)
        {
            if (fit->is_unbounded())
                continue;

            Polyline2_t<Scalar> cell = extract_voronoi_cell<Scalar>(fit);

            if (cell.points.size() >= 3)
                result.voronoi_cells.push_back(std::move(cell));
        }
        return result;
    }

    template <typename Scalar>
    inline Eigen::AlignedBox<Scalar, 2> computeAABB(const std::vector<Polyline2_t<Scalar>> &polylines)
    {
        Eigen::AlignedBox<Scalar, 2> box;
        box.setEmpty();

        for (const auto &poly : polylines)
        {
            for (const auto &p : poly.points)
            {
                box.extend(p);
            }
        }
        return box;
    }

    int largestRectangleInHistogramPoints(const std::vector<int> &h, int &left, int &right);

    template <typename Scalar>
    Polyline2_t<Scalar> getMaxRectInPoly(const Polyline2_t<Scalar> &poly, double gridSize)
    {
        // ---- Step 1: AABB ----
        Eigen::AlignedBox<Scalar, 2> aabb = poly.getAABB2();
        if (aabb.isEmpty())
            return Polyline2_t<Scalar>();

        Scalar minx = aabb.min().x();
        Scalar miny = aabb.min().y();
        Scalar maxx = aabb.max().x();
        Scalar maxy = aabb.max().y();

        Scalar width = maxx - minx;
        Scalar height = maxy - miny;

        int rows = std::max(1, int(std::ceil(width / gridSize)));
        int cols = std::max(1, int(std::ceil(height / gridSize)));

        Scalar cellX = width / rows;
        Scalar cellY = height / cols;

        // ---- Step 2: NodeInside (points) ----
        std::vector<std::vector<int>> nodeInside(
            rows + 1, std::vector<int>(cols + 1, 0));

        for (int r = 0; r <= rows; ++r)
        {
            for (int c = 0; c <= cols; ++c)
            {
                Scalar x = minx + r * cellX;
                Scalar y = miny + c * cellY;
                nodeInside[r][c] = util::Math2<Scalar>::point_in_poly(poly.points, Vector2<Scalar>(x, y)) ? 1 : 0;
            }
        }

        // ---- Step 3: CellInside (FOUR corners) ----
        std::vector<std::vector<int>> cellInside(
            rows, std::vector<int>(cols, 0));

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                cellInside[r][c] = nodeInside[r][c] & nodeInside[r + 1][c] & nodeInside[r][c + 1] & nodeInside[r + 1][c + 1];
            }
        }

        // ---- Step 4: Histogram + stack ----
        std::vector<int> hist(cols, 0);

        int bestArea = 0;
        int bestR0 = 0, bestR1 = -1;
        int bestC0 = 0, bestC1 = -1;

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
                hist[c] = cellInside[r][c] ? hist[c] + 1 : 0;

            std::vector<int> st;
            int c = 0;
            while (c < cols)
            {
                if (st.empty() || hist[st.back()] <= hist[c])
                {
                    st.push_back(c++);
                }
                else
                {
                    int top = st.back();
                    st.pop_back();
                    int width = st.empty() ? c : (c - st.back() - 1);
                    int height = hist[top];
                    int area = width * height;

                    if (area > bestArea)
                    {
                        bestArea = area;
                        bestR0 = r - height + 1;
                        bestR1 = r;
                        bestC0 = st.empty() ? 0 : (st.back() + 1);
                        bestC1 = c - 1;
                    }
                }
            }

            while (!st.empty())
            {
                int top = st.back();
                st.pop_back();
                int width = st.empty() ? c : (c - st.back() - 1);
                int height = hist[top];
                int area = width * height;

                if (area > bestArea)
                {
                    bestArea = area;
                    bestR0 = r - height + 1;
                    bestR1 = r;
                    bestC0 = st.empty() ? 0 : (st.back() + 1);
                    bestC1 = c - 1;
                }
            }
        }

        if (bestArea <= 0)
            return Polyline2_t<Scalar>();

        // ---- Step 5: Cell indices → world coordinates ----
        Scalar x0 = minx + bestR0 * cellX;
        Scalar y0 = miny + bestC0 * cellY;
        Scalar x1 = minx + (bestR1 + 1) * cellX;
        Scalar y1 = miny + (bestC1 + 1) * cellY;

        std::vector<Vector2<Scalar>> rect = {
            {x0, y0},
            {x1, y0},
            {x1, y1},
            {x0, y1},
            {x0, y0}};

        return Polyline2_t<Scalar>(rect, true);
    }

    template <typename Scalar>
    Polyline2_t<Scalar> getMaxRectInPolyWithGrid(
        const Polyline2_t<Scalar> &poly,
        double gridSize,
        std::vector<Polyline2_t<Scalar>> &insideGrids, // 输出：所有内部单元格（四个顶点均在多边形内）
        std::vector<Vector2<Scalar>> &gridPoints)      // 输出：所有网格顶点
    {
        insideGrids.clear();
        gridPoints.clear();
        // ---- Step 1: AABB ----
        Eigen::AlignedBox<Scalar, 2> aabb = poly.getAABB2();
        if (aabb.isEmpty())
            return Polyline2_t<Scalar>();

        Scalar minx = aabb.min().x();
        Scalar miny = aabb.min().y();
        Scalar maxx = aabb.max().x();
        Scalar maxy = aabb.max().y();

        Scalar width = maxx - minx;
        Scalar height = maxy - miny;

        int rows = std::max(1, int(std::ceil(width / gridSize)));
        int cols = std::max(1, int(std::ceil(height / gridSize)));

        Scalar cellX = width / rows;
        Scalar cellY = height / cols;

        // ---- Step 2: NodeInside (points) ----
        std::vector<std::vector<int>> nodeInside(
            rows + 1, std::vector<int>(cols + 1, 0));

        // 预先分配 gridPoints 大小
        gridPoints.clear();
        gridPoints.reserve((rows + 1) * (cols + 1));

        for (int r = 0; r <= rows; ++r)
        {
            for (int c = 0; c <= cols; ++c)
            {
                Scalar x = minx + r * cellX;
                Scalar y = miny + c * cellY;
                Vector2<Scalar> pt(x, y);
                gridPoints.push_back(pt);
                nodeInside[r][c] = util::Math2<Scalar>::point_in_poly(poly.points, pt) ? 1 : 0;
            }
        }

        // ---- Step 3: CellInside (FOUR corners) 及收集 insideGrids ----
        insideGrids.clear();
        insideGrids.reserve(rows * cols);

        std::vector<std::vector<int>> cellInside(rows, std::vector<int>(cols, 0));

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                bool inside = nodeInside[r][c] && nodeInside[r + 1][c] &&
                              nodeInside[r][c + 1] && nodeInside[r + 1][c + 1];
                cellInside[r][c] = inside ? 1 : 0;

                if (inside)
                {
                    // 构建该单元格的四个角点（顺时针或逆时针均可，此处按顺序）
                    Vector2<Scalar> p0(minx + r * cellX, miny + c * cellY);
                    Vector2<Scalar> p1(minx + (r + 1) * cellX, miny + c * cellY);
                    Vector2<Scalar> p2(minx + (r + 1) * cellX, miny + (c + 1) * cellY);
                    Vector2<Scalar> p3(minx + r * cellX, miny + (c + 1) * cellY);
                    std::vector<Vector2<Scalar>> cellPoly = {p0, p1, p2, p3, p0};
                    insideGrids.emplace_back(cellPoly, true);
                }
            }
        }

        // ---- Step 4: Histogram + stack (寻找最大内接矩形) ----
        std::vector<int> hist(cols, 0);

        int bestArea = 0;
        int bestR0 = 0, bestR1 = -1;
        int bestC0 = 0, bestC1 = -1;

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
                hist[c] = cellInside[r][c] ? hist[c] + 1 : 0;

            std::vector<int> st;
            int c = 0;
            while (c < cols)
            {
                if (st.empty() || hist[st.back()] <= hist[c])
                {
                    st.push_back(c++);
                }
                else
                {
                    int top = st.back();
                    st.pop_back();
                    int width = st.empty() ? c : (c - st.back() - 1);
                    int height = hist[top];
                    int area = width * height;

                    if (area > bestArea)
                    {
                        bestArea = area;
                        bestR0 = r - height + 1;
                        bestR1 = r;
                        bestC0 = st.empty() ? 0 : (st.back() + 1);
                        bestC1 = c - 1;
                    }
                }
            }

            while (!st.empty())
            {
                int top = st.back();
                st.pop_back();
                int width = st.empty() ? c : (c - st.back() - 1);
                int height = hist[top];
                int area = width * height;

                if (area > bestArea)
                {
                    bestArea = area;
                    bestR0 = r - height + 1;
                    bestR1 = r;
                    bestC0 = st.empty() ? 0 : (st.back() + 1);
                    bestC1 = c - 1;
                }
            }
        }

        if (bestArea <= 0)
            return Polyline2_t<Scalar>();

        // ---- Step 5: Cell indices → world coordinates ----
        Scalar x0 = minx + bestR0 * cellX;
        Scalar y0 = miny + bestC0 * cellY;
        Scalar x1 = minx + (bestR1 + 1) * cellX;
        Scalar y1 = miny + (bestC1 + 1) * cellY;

        std::vector<Vector2<Scalar>> rect = {
            {x0, y0},
            {x1, y0},
            {x1, y1},
            {x0, y1},
            {x0, y0}};

        return Polyline2_t<Scalar>(rect, true);
    }

    template <typename Scalar>
    inline Eigen::Matrix<Scalar, 2, 2> rotationToXAxis(const Eigen::Vector2<Scalar> &axis)
    {
        Eigen::Vector2<Scalar> a = axis.normalized();
        // a -> (1,0)
        return (Eigen::Matrix<Scalar, 2, 2>() << a.x(), a.y(), -a.y(), a.x()).finished();
    }

    template <typename Scalar>
    Polyline2_t<Scalar> rotatePoly(const Polyline2_t<Scalar> &poly, const Eigen::Matrix<Scalar, 2, 2> &R)
    {
        std::vector<Vector2<Scalar>> pts;
        pts.reserve(poly.points.size());

        for (const auto &p : poly.points)
            pts.emplace_back(R * p);

        return Polyline2_t<Scalar>(pts, poly.isClosed);
    }

    template <typename Scalar>
    inline Eigen::Matrix<Scalar, 2, 2> rotationMatrixFromDegree(double degree)
    {
        // degree -> radian
        Scalar rad = Scalar(degree * util::Litten_PI / Scalar(180));

        Scalar c = std::cos(rad);
        Scalar s = std::sin(rad);

        Eigen::Matrix<Scalar, 2, 2> R;
        R << c, -s,
            s, c;
        return R;
    }

    template <typename Scalar>
    Polyline2_t<Scalar> rotatePolyByDegree(const Polyline2_t<Scalar> &poly, double degree)
    {
        Eigen::Matrix<Scalar, 2, 2> R = rotationMatrixFromDegree<Scalar>(degree);

        std::vector<Vector2<Scalar>> pts;
        pts.reserve(poly.points.size());

        for (const auto &p : poly.points)
            pts.emplace_back(R * p);

        return Polyline2_t<Scalar>(pts, poly.isClosed);
    }

    template <typename Scalar>
    std::vector<Vector2<Scalar>>
    samplePointsOnPolygonWithSpacing(const Polyline2_t<Scalar> &poly,int n,unsigned seed = 0){
        std::vector<Vector2<Scalar>> result;
        if (poly.points.size() < 2 || n <= 0)
            return result;

        // ---------- total length ----------
        Scalar total_length = 0;
        std::vector<Scalar> seg_lengths;
        seg_lengths.reserve(poly.points.size() - 1);

        for (size_t i = 0; i + 1 < poly.points.size(); ++i)
        {
            Scalar len = (poly.points[i + 1] - poly.points[i]).norm();
            seg_lengths.push_back(len);
            total_length += len;
        }

        if (total_length <= Scalar(1e-8))
            return result;

        // ---------- spacing control ----------
        Scalar target = total_length / Scalar(n);
        Scalar range = total_length / (Scalar(3) * Scalar(n)); // ★ 改这里

        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<Scalar> jitter(-range, range);

        // ---------- arc-length positions ----------
        std::vector<Scalar> arc_pos;
        arc_pos.reserve(n);

        Scalar s = 0;
        for (int i = 0; i < n; ++i)
        {
            Scalar step = std::max(Scalar(0), target + jitter(rng));
            s += step;

            // wrap
            if (s >= total_length)
                s = std::fmod(s, total_length);

            arc_pos.push_back(s);
        }

        std::sort(arc_pos.begin(), arc_pos.end());

        // ---------- arc-length → point ----------
        for (Scalar a : arc_pos)
        {
            Scalar acc = 0;
            for (size_t i = 0; i < seg_lengths.size(); ++i)
            {
                if (acc + seg_lengths[i] >= a)
                {
                    Scalar t = (a - acc) / seg_lengths[i];
                    result.push_back(
                        poly.points[i] * (Scalar(1) - t) +
                        poly.points[i + 1] * t);
                    break;
                }
                acc += seg_lengths[i];
            }
        }

        return result;
    }
}