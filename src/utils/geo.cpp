#include "geo.h"

namespace geo
{

    PointKey makeKey(const Eigen::Vector3f &p, float eps)
    {
        return {
            static_cast<int>(std::round(p.x() / eps)),
            static_cast<int>(std::round(p.y() / eps)),
            static_cast<int>(std::round(p.z() / eps))};
    }

    std::vector<Polyline3> buildPolylines(const std::vector<geo::Segment> &segments, float eps)
    {
        using Key = PointKey;

        std::unordered_map<Key, std::vector<Edge>, PointKeyHash> adjacency;

        // ===== 1. 构建端点邻接 =====
        for (int i = 0; i < (int)segments.size(); ++i)
        {
            const auto &s = segments[i];

            adjacency[makeKey(s.p0, eps)].push_back({i, 0});
            adjacency[makeKey(s.p1, eps)].push_back({i, 1});
        }

        std::vector<bool> used(segments.size(), false);
        std::vector<Polyline3> polylines;

        // ===== 2. 从端点（度=1）开始，生成开曲线 =====
        for (auto &[key, edges] : adjacency)
        {
            if (edges.size() != 1)
                continue;

            int segId = edges[0].segId;
            if (used[segId])
                continue;

            Polyline3 pl;
            followChain(segId, edges[0].end,
                        segments, adjacency, used, eps, pl);
            polylines.push_back(std::move(pl));
        }

        // ===== 3. 剩余的是闭合曲线 =====
        for (int i = 0; i < (int)segments.size(); ++i)
        {
            if (used[i])
                continue;

            Polyline3 pl;
            followChain(i, 0,
                        segments, adjacency, used, eps, pl);
            pl.closed = true;
            polylines.push_back(std::move(pl));
        }

        return polylines;
    }

    void followChain(
        int startSeg,
        int startEnd,
        const std::vector<geo::Segment> &segments,
        const std::unordered_map<PointKey, std::vector<Edge>, PointKeyHash> &adj,
        std::vector<bool> &used,
        float eps,
        Polyline3 &out)
    {
        int currSeg = startSeg;
        int currEnd = startEnd;

        Eigen::Vector3f currPt =
            (currEnd == 0) ? segments[currSeg].p0
                           : segments[currSeg].p1;

        out.points.push_back(currPt);

        while (true)
        {
            used[currSeg] = true;

            // 走到 segment 的另一端
            Eigen::Vector3f nextPt =
                (currEnd == 0) ? segments[currSeg].p1
                               : segments[currSeg].p0;

            out.points.push_back(nextPt);

            PointKey key = makeKey(nextPt, eps);
            auto it = adj.find(key);
            if (it == adj.end())
                break;

            // 寻找下一条未使用的 segment
            int nextSeg = -1;
            int nextEnd = -1;

            for (const auto &e : it->second)
            {
                if (e.segId != currSeg && !used[e.segId])
                {
                    nextSeg = e.segId;
                    nextEnd = e.end;
                    break;
                }
            }

            if (nextSeg == -1)
                break;

            currSeg = nextSeg;
            currEnd = nextEnd;
        }
    }

    void PolygonMesh::build()
    {
        // std::cout << "\n[BUILD] begin\n";

        if (points.size() < 3)
        {
            std::cout << "[BUILD ERROR] points < 3\n";
            return;
        }

        using Vector3 = Eigen::Vector3f;
        using Polyloop3 = polyloop::Polyloop3;

        Polyloop3 poly(points);

        if (poly.triangles().empty())
        {
            std::cout << "[BUILD ERROR] triangulation failed\n";
            return;
        }

        

        const Eigen::Vector3f normal = poly.normal();

        if (normal.norm() < 1e-6f)
        {
            std::cout << "[BUILD ERROR] invalid normal\n";
            return;
        }

        const Eigen::Vector3f offset = normal * height;

        const uint32_t n = static_cast<uint32_t>(points.size());

        mesh.vertices.clear();
        mesh.indices.clear();

        mesh.vertices.reserve(n * 2);
        mesh.indices.reserve(poly.triangles().size() * 6 + n * 6);

        // ===== 顶面 =====
        for (uint32_t i = 0; i < n; ++i)
        {
            Vertex v;
            v.position = points[i] + offset;
            v.normal = normal;
            mesh.vertices.push_back(v);
        }

        // ===== 底面 =====
        for (uint32_t i = 0; i < n; ++i)
        {
            Vertex v;
            v.position = points[i];
            v.normal = -normal;
            mesh.vertices.push_back(v);
        }

        // ===== 顶 / 底 =====
        for (auto &tri : poly.triangles())
        {
            uint32_t i0 = tri[0];
            uint32_t i1 = tri[1];
            uint32_t i2 = tri[2];

            // 防越界
            if (i0 >= n || i1 >= n || i2 >= n)
            {
                std::cout << "[BUILD ERROR] triangle index out of range\n";
                continue;
            }

            mesh.indices.push_back(i0);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);

            mesh.indices.push_back(i2 + n);
            mesh.indices.push_back(i1 + n);
            mesh.indices.push_back(i0 + n);
        }

        // ===== 侧面 =====
        for (uint32_t i = 0; i < n; ++i)
        {
            uint32_t j = (i + 1) % n;

            uint32_t top0 = i;
            uint32_t top1 = j;
            uint32_t bot0 = i + n;
            uint32_t bot1 = j + n;

            const Eigen::Vector3f &p0 = mesh.vertices[top0].position;
            const Eigen::Vector3f &p1 = mesh.vertices[top1].position;

            Eigen::Vector3f edge = p1 - p0;

            Eigen::Vector3f sideNormal = edge.cross(normal);

            if (sideNormal.norm() < 1e-6f)
            {
                sideNormal = normal; // fallback
            }
            else
            {
                sideNormal.normalize();
            }

            mesh.indices.push_back(top0);
            mesh.indices.push_back(bot0);
            mesh.indices.push_back(bot1);

            mesh.indices.push_back(top0);
            mesh.indices.push_back(bot1);
            mesh.indices.push_back(top1);

            mesh.vertices[top0].normal = sideNormal;
            mesh.vertices[top1].normal = sideNormal;
            mesh.vertices[bot0].normal = sideNormal;
            mesh.vertices[bot1].normal = sideNormal;
        }

        // std::cout << "[BUILD] vertex count: " << mesh.vertices.size() << "\n";
        // std::cout << "[BUILD] index count: " << mesh.indices.size() << "\n";
    }

    Mesh buildRaylibMesh(const MeshData &src)
    {
        // std::cout << "\n[UPLOAD] begin\n";
        // std::cout << "[UPLOAD] vertex count: " << src.vertices.size() << "\n";
        // std::cout << "[UPLOAD] index count: " << src.indices.size() << "\n";

        Mesh mesh = {0};

        // ===== 基础检查 =====
        if (src.vertices.empty() || src.indices.empty())
        {
            std::cout << "[UPLOAD ERROR] empty mesh data\n";
            return mesh;
        }

        if (src.indices.size() % 3 != 0)
        {
            std::cout << "[UPLOAD ERROR] indices not multiple of 3\n";
            return mesh;
        }

        if (src.vertices.size() > 65535)
        {
            std::cout << "[UPLOAD ERROR] vertex exceeds 65535 (uint16 limit)\n";
            return mesh;
        }

        // ===== index 越界检查（🔥关键）=====
        for (size_t i = 0; i < src.indices.size(); ++i)
        {
            if (src.indices[i] >= src.vertices.size())
            {
                std::cout << "[UPLOAD ERROR] index out of range at " << i
                          << " value=" << src.indices[i] << "\n";
                return mesh;
            }
        }

        mesh.vertexCount = (int)src.vertices.size();
        mesh.triangleCount = (int)src.indices.size() / 3;

        mesh.vertices = (float *)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
        mesh.normals = (float *)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
        mesh.indices = (unsigned short *)MemAlloc(mesh.triangleCount * 3 * sizeof(unsigned short));
        mesh.colors = (unsigned char *)MemAlloc(mesh.vertexCount * 4 * sizeof(unsigned char));

        // ===== 顶点 =====
        for (int i = 0; i < mesh.vertexCount; ++i)
        {
            const auto &v = src.vertices[i];

            if (!std::isfinite(v.position.x()) ||
                !std::isfinite(v.position.y()) ||
                !std::isfinite(v.position.z()))
            {
                std::cout << "[UPLOAD ERROR] NaN vertex at " << i << "\n";
                return Mesh{};
            }

            mesh.vertices[i * 3 + 0] = v.position.x();
            mesh.vertices[i * 3 + 1] = v.position.z();
            mesh.vertices[i * 3 + 2] = -v.position.y();

            mesh.normals[i * 3 + 0] = v.normal.x();
            mesh.normals[i * 3 + 1] = v.normal.z();
            mesh.normals[i * 3 + 2] = -v.normal.y();

            mesh.colors[i * 4 + 0] = 255;
            mesh.colors[i * 4 + 1] = 255;
            mesh.colors[i * 4 + 2] = 255;
            mesh.colors[i * 4 + 3] = 255;
        }

        for (size_t i = 0; i < src.indices.size(); ++i)
        {
            mesh.indices[i] = (unsigned short)src.indices[i];
        }

        UploadMesh(&mesh, true);

        // std::cout << "[UPLOAD] success\n";
        return mesh;
    }

    void PolygonMesh::upload()
    {
        // std::cout << "[UPLOAD] model begin\n";

        Mesh m = buildRaylibMesh(mesh);

        if (m.vertexCount == 0)
        {
            std::cout << "[UPLOAD ERROR] invalid mesh, skip LoadModel\n";
            return;
        }

        // std::cout << "[UPLOAD] calling LoadModelFromMesh...\n";

        model = LoadModelFromMesh(m);

        // std::cout << "[UPLOAD] model created SUCCESS\n";
    }

    void PolygonMesh::draw(Color color, float colorAlpha, bool outline, bool wireframe, float wireframeAlpha)const
    {
        // 绘制模型
        if (model.meshCount == 0)
            return;
        DrawModel(model, {0, 0, 0}, 1.0f, Fade(color, colorAlpha));

        if (outline)
        {
            const uint32_t n = static_cast<uint32_t>(mesh.vertices.size());
            const uint32_t half = n / 2;

            auto toRay = [](const Eigen::Vector3f &p)
            {
                return Vector3{p.x(), p.z(), -p.y()};
            };

            for (uint32_t i = 0; i < half; ++i)
            {
                uint32_t j = (i + 1) % half;
                // ===== 1. 顶面轮廓 =====
                DrawLine3D(
                    toRay(mesh.vertices[i].position),
                    toRay(mesh.vertices[j].position),
                    RL_BLACK);
                // ===== 2. 底面轮廓 =====
                DrawLine3D(
                    toRay(mesh.vertices[i + half].position),
                    toRay(mesh.vertices[j + half].position),
                    RL_BLACK);
                // ===== 3. 竖向轮廓边 =====
                DrawLine3D(
                    toRay(mesh.vertices[i].position),
                    toRay(mesh.vertices[i + half].position),
                    RL_BLACK);
            }
        }
        if (wireframe)
        {
            // 绘制线框
            DrawModelWires(model, {0, 0, 0}, 1.0f, Fade(RL_BLACK, wireframeAlpha));
        }
    }

    void PolygonMesh::regenerate(float newHeight)
    {
        height = newHeight;

        build();

        if (model.meshCount > 0)
        {
            std::cout << "[REGEN] unloading old model\n";
            UnloadModel(model);
        }

        upload();
    }

    int largestRectangleInHistogramPoints(
        const std::vector<int> &h,
        int &left, int &right)
    {
        std::vector<int> st;
        int maxArea = 0;
        int n = h.size();

        for (int i = 0; i <= n; ++i)
        {
            int cur = (i == n) ? 0 : h[i];
            while (!st.empty() && cur < h[st.back()])
            {
                int height = h[st.back()];
                st.pop_back();
                int l = st.empty() ? 0 : st.back() + 1;
                int r = i - 1;

                int area = height * (r - l + 1);
                if (area > maxArea)
                {
                    maxArea = area;
                    left = l;
                    right = r;
                }
            }
            st.push_back(i);
        }
        return maxArea;
    }

}