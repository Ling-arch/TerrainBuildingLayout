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
        if (points.size() < 3)
            return;
        using Vector3 = Eigen::Vector3f;
        using Polyloop3 = polyloop::Polyloop3;

        // =====  使用已有的 Polyloop3 =====
        Polyloop3 poly(points);
        if (poly.triangles().empty())
            return;

        const Eigen::Vector3f normal = poly.normal();
        const Eigen::Vector3f offset = normal * height;

        const uint32_t n = static_cast<uint32_t>(points.size());
        mesh.vertices.clear();
        mesh.indices.clear();
        // ===== 2. 顶点：顶面 + 底面 =====
        mesh.vertices.reserve(n * 2);
        mesh.indices.reserve(poly.triangles().size() * 6 + n * 6);

        // --- 顶面 ---
        for (uint32_t i = 0; i < n; ++i)
        {
            Vertex v;
            v.position = points[i] + offset;
            v.normal = normal;
            mesh.vertices.push_back(v);
        }

        // --- 底面 ---
        for (uint32_t i = 0; i < n; ++i)
        {
            Vertex v;
            v.position = points[i];
            v.normal = -normal;
            mesh.vertices.push_back(v);
        }

        // ===== 3. 顶 / 底面索引 =====
        // Polyloop3::triangles_ 是二维三角化索引
        for (auto &tri : poly.triangles())
        {
            uint32_t i0 = tri[0];
            uint32_t i1 = tri[1];
            uint32_t i2 = tri[2];

            // 顶面（CCW）
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);

            // 底面（反向，看向下）
            mesh.indices.push_back(i2 + n);
            mesh.indices.push_back(i1 + n);
            mesh.indices.push_back(i0 + n);
        }

        // ===== 4. 侧面 =====
        // 每条边 -> 一个 quad -> 2 个三角形
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
            Eigen::Vector3f sideNormal = edge.cross(normal).normalized();

            // 第一个三角形
            mesh.indices.push_back(top0);
            mesh.indices.push_back(bot0);
            mesh.indices.push_back(bot1);

            // 第二个三角形
            mesh.indices.push_back(top0);
            mesh.indices.push_back(bot1);
            mesh.indices.push_back(top1);

            // 覆盖侧面法线（硬边）
            mesh.vertices[top0].normal = sideNormal;
            mesh.vertices[top1].normal = sideNormal;
            mesh.vertices[bot0].normal = sideNormal;
            mesh.vertices[bot1].normal = sideNormal;
        }
    }

    Mesh buildRaylibMesh(const MeshData &src)
    {

        Mesh mesh = {0};

        mesh.vertexCount = (int)src.vertices.size();
        mesh.triangleCount = (int)src.indices.size() / 3;

        // ===== 分配内存 =====
        mesh.vertices = (float *)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
        mesh.normals = (float *)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
        mesh.indices = (unsigned short *)MemAlloc(mesh.triangleCount * 3 * sizeof(unsigned short));
        mesh.colors = (unsigned char *)MemAlloc(mesh.vertexCount * 4 * sizeof(unsigned char));

        // ===== 顶点 & 法线 =====
        for (int i = 0; i < mesh.vertexCount; ++i)
        {
            const auto &v = src.vertices[i];

            mesh.vertices[i * 3 + 0] = v.position.x();
            mesh.vertices[i * 3 + 1] = v.position.z();
            mesh.vertices[i * 3 + 2] = -v.position.y();

            mesh.normals[i * 3 + 0] = v.normal.x();
            mesh.normals[i * 3 + 1] = v.normal.z();
            mesh.normals[i * 3 + 2] = -v.normal.y();

            // 默认白色
            mesh.colors[i * 4 + 0] = 255;
            mesh.colors[i * 4 + 1] = 255;
            mesh.colors[i * 4 + 2] = 255;
            mesh.colors[i * 4 + 3] = 255;
        }

        // ===== 索引 =====
        for (size_t i = 0; i < src.indices.size(); ++i)
        {
            mesh.indices[i] = (unsigned short)src.indices[i];
        }

        // ===== 上传到 GPU =====
        UploadMesh(&mesh, true);

        return mesh;
    }

    void PolygonMesh::upload()
    {
        model = LoadModelFromMesh(buildRaylibMesh(mesh));
    }

    void PolygonMesh::draw(Color color, float colorAlpha, bool outline, bool wireframe, float wireframeAlpha)
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
            UnloadModel(model);
        upload();
    }


    
}