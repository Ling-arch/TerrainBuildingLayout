#include "terrain.h"
#include <cmath>

namespace terrain
{
    // 一个简单的 2D 噪声（你可以以后换 Perlin / Simplex）
    float noise2D(int x, int y)
    {
        int n = x + y * 57;
        n = (n << 13) ^ n;
        return 1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f;
    }

    Terrain::Terrain(int w, int h, float cs)
        : width(w), height(h), cellSize(cs)
    {
        heightmap.resize((width + 1) * (height + 1), 0.f);
    }

    Terrain::Terrain(int w, int h, float cs, float frequency, float amplitude)
        : width(w), height(h), cellSize(cs)
    {
        heightmap.resize((width + 1) * (height + 1), 0.f);
        generateHeight(frequency, amplitude);
        buildMesh();
        upload();
    }

    void Terrain::generateHeight(float freq, float amp)
    {
        for (int y = 0; y <= height; y++)
        {
            for (int x = 0; x <= width; x++)
            {
                float nx = x * freq;
                float ny = y * freq;

                heightmap[y * (width + 1) + x] = fbm2D(nx, ny, 2, 1.6f, 0.7f) * amp;
            }
        }
    }

    void Terrain::upload()
    {
        Mesh raylib_mesh = buildRaylibMesh(mesh);
        model = LoadModelFromMesh(raylib_mesh);
    }

    void Terrain::draw() const
    {
        switch (viewMode)
        {
        case TerrainViewMode::Wire:
            DrawModelWires(model, {0, 0, 0}, 1.0f, BLACK);
            break;

        case TerrainViewMode::Aspect:
        case TerrainViewMode::Slope:
            DrawModel(model, {0, 0, 0}, 1.0f, WHITE);
            break;

        case TerrainViewMode::Lit:
        default:
            DrawModel(model, {0, 0, 0}, 1.0f, GRAY);
            break;
        }
    }

    void Terrain::applyFaceColor(TerrainViewMode mode)
    {
        Mesh &m = model.meshes[0];

        // 1. 确保有 colors buffer
        if (!m.colors)
        {
            m.colors = (unsigned char *)MemAlloc(m.vertexCount * 4);
        }

        // 2. 默认底色
        for (int i = 0; i < m.vertexCount; ++i)
        {
            m.colors[i * 4 + 0] = 200;
            m.colors[i * 4 + 1] = 200;
            m.colors[i * 4 + 2] = 200;
            m.colors[i * 4 + 3] = 255;
        }

        // 3. 按 face 着色
        for (size_t f = 0; f < faceInfos.size(); ++f)
        {
            const auto &face = faceInfos[f];

            Color c{180, 180, 180, 255};

            // 平地判断
            bool isFlat = face.slope < 1e-4f;

            if (!isFlat)
            {
                if (mode == TerrainViewMode::Aspect)
                {
                    // aspect ∈ [0, 2π)
                    c = renderUtil::AspectToColor(face.aspect);
                }
                else if (mode == TerrainViewMode::Slope)
                {
                    // slope ∈ [0, π/2]
                    float t = face.slope / (PI * 0.5f);
                    t = std::clamp(t, 0.0f, 1.0f);

                    // 蓝 → 红（缓坡 → 陡坡）
                    c = {
                        (unsigned char)(255 * t),
                        0,
                        (unsigned char)(255 * (1.0f - t)),
                        255};
                }
            }

            int i0 = mesh.indices[f * 3 + 0];
            int i1 = mesh.indices[f * 3 + 1];
            int i2 = mesh.indices[f * 3 + 2];

            auto setColor = [&](int vi)
            {
                m.colors[vi * 4 + 0] = c.r;
                m.colors[vi * 4 + 1] = c.g;
                m.colors[vi * 4 + 2] = c.b;
                m.colors[vi * 4 + 3] = 255;
            };

            setColor(i0);
            setColor(i1);
            setColor(i2);
        }

        // 4. 上传到 GPU
        UpdateMeshBuffer(m, 3, m.colors, m.vertexCount * 4, 0);
    }

    void Terrain::setViewMode(TerrainViewMode mode)
    {
        viewMode = mode;

        switch (viewMode)
        {
        case TerrainViewMode::Aspect:
            applyFaceColor(mode);
            break;

        case TerrainViewMode::Slope:
            applyFaceColor(mode);
            break;

        case TerrainViewMode::Lit:
            break;

        case TerrainViewMode::Wire:
            break;
        }
    }

    const std::vector<Vector3> Terrain::getMeshVertices()
    {
        std::vector<Vector3> vertices;
        vertices.reserve(mesh.vertices.size());
        for (auto &v : mesh.vertices)
        {
            vertices.push_back({v.position.x(), v.position.z() + 0.5f, -v.position.y()});
        }
        return vertices;
    }

    void Terrain::buildMesh()
    {
        mesh.vertices.clear();
        mesh.indices.clear();
        faceInfos.clear();
        mesh.vertices.reserve((height + 1) * (width + 1));
        mesh.indices.reserve(height * width * 6);
        faceInfos.reserve(height * width * 2);
        mesh.gridWidth = width;
        mesh.gridHeight = height;

        minHeight = std::numeric_limits<float>::max();
        maxHeight = std::numeric_limits<float>::lowest();
        // 顶点
        for (int y = 0; y <= height; ++y)
        {
            for (int x = 0; x <= width; ++x)
            {
                float h = heightmap[vertexIndex(x, y)];
                minHeight = std::min(minHeight, h);
                maxHeight = std::max(maxHeight, h);
                TerrainVertex v;
                v.position = Eigen::Vector3f(x * cellSize - width / 2.f, y * cellSize - height / 2.f, h);

                v.normal = Eigen::Vector3f::Zero();

                mesh.vertices.push_back(v);
            }
        }

        // 索引（右手 CCW）
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int i0 = vertexIndex(x, y);
                int i1 = vertexIndex(x + 1, y);
                int i2 = vertexIndex(x, y + 1);
                int i3 = vertexIndex(x + 1, y + 1);

                mesh.indices.insert(mesh.indices.end(), {(uint32_t)i0, (uint32_t)i1, (uint32_t)i2, (uint32_t)i1, (uint32_t)i3, (uint32_t)i2});
            }
        }
        for (size_t i = 0; i < mesh.indices.size(); i += 3)
        {
            int i0 = mesh.indices[i + 0];
            int i1 = mesh.indices[i + 1];
            int i2 = mesh.indices[i + 2];

            const auto &v0 = mesh.vertices[i0].position;
            const auto &v1 = mesh.vertices[i1].position;
            const auto &v2 = mesh.vertices[i2].position;

            Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
            float area = n.norm();
            n.normalize();

            // 坡向（使用下坡方向）
            Eigen::Vector3f down = -n;
            Eigen::Vector3f h(down.x(), down.y(), 0.0f);

            float aspect = 0.f;
            if (h.norm() > 1e-5f)
            {
                aspect = atan2(h.y(), h.x());
                if (aspect < 0)
                    aspect += 2.0f * PI;
            }

            float slope = acos(std::clamp(n.z(), -1.0f, 1.0f));

            faceInfos.push_back({n, slope, aspect});

            mesh.vertices[i0].normal += n * area;
            mesh.vertices[i1].normal += n * area;
            mesh.vertices[i2].normal += n * area;
        }

        // ===== 4. 归一化顶点法线 =====
        for (auto &v : mesh.vertices)
        {
            if (v.normal.norm() > 1e-6f)
                v.normal.normalize();
            else
                v.normal = Eigen::Vector3f(0, 0, 1);
        }
    }

    std::vector<geo::Segment> Terrain::extractContourAtHeight(float isoHeight) const
    {
        std::vector<geo::Segment> segments;

        for (size_t i = 0; i < mesh.indices.size(); i += 3)
        {
            int i0 = mesh.indices[i + 0];
            int i1 = mesh.indices[i + 1];
            int i2 = mesh.indices[i + 2];

            const auto &p0 = mesh.vertices[i0].position;
            const auto &p1 = mesh.vertices[i1].position;
            const auto &p2 = mesh.vertices[i2].position;

            float h0 = p0.z();
            float h1 = p1.z();
            float h2 = p2.z();

            bool b0 = h0 > isoHeight;
            bool b1 = h1 > isoHeight;
            bool b2 = h2 > isoHeight;

            int count = (int)b0 + (int)b1 + (int)b2;
            if (count == 0 || count == 3)
                continue;

            auto interp = [&](const Eigen::Vector3f &a,
                              const Eigen::Vector3f &b,
                              float ha, float hb)
            {
                float t = (isoHeight - ha) / (hb - ha);
                return a + t * (b - a);
            };

            std::vector<Eigen::Vector3f> pts;

            if (b0 != b1)
                pts.push_back(interp(p0, p1, h0, h1));
            if (b1 != b2)
                pts.push_back(interp(p1, p2, h1, h2));
            if (b2 != b0)
                pts.push_back(interp(p2, p0, h2, h0));

            if (pts.size() == 2)
            {
                segments.push_back({pts[0], pts[1]});
            }
        }

        return segments;
    }

    std::vector<ContourLayer> Terrain::extractContours(float gap) const
    {
        std::unordered_map<int, ContourLayer> layerMap;

        for (size_t i = 0; i < mesh.indices.size(); i += 3)
        {
            int i0 = mesh.indices[i + 0];
            int i1 = mesh.indices[i + 1];
            int i2 = mesh.indices[i + 2];

            const auto &p0 = mesh.vertices[i0].position;
            const auto &p1 = mesh.vertices[i1].position;
            const auto &p2 = mesh.vertices[i2].position;

            float h0 = p0.z();
            float h1 = p1.z();
            float h2 = p2.z();

            float hmin = std::min({h0, h1, h2});
            float hmax = std::max({h0, h1, h2});

            int k0 = (int)std::ceil(hmin / gap);
            int k1 = (int)std::floor(hmax / gap);

            if (k0 > k1)
                continue;

            for (int k = k0; k <= k1; ++k)
            {
                float isoHeight = k * gap;

                bool b0 = h0 > isoHeight;
                bool b1 = h1 > isoHeight;
                bool b2 = h2 > isoHeight;

                int count = (int)b0 + (int)b1 + (int)b2;
                if (count == 0 || count == 3)
                    continue;

                auto interp = [&](const Eigen::Vector3f &a,
                                  const Eigen::Vector3f &b,
                                  float ha, float hb)
                {
                    float t = (isoHeight - ha) / (hb - ha);
                    return a + t * (b - a);
                };

                Eigen::Vector3f pts[2];
                int ptCount = 0;

                if (b0 != b1)
                    pts[ptCount++] = interp(p0, p1, h0, h1);
                if (b1 != b2)
                    pts[ptCount++] = interp(p1, p2, h1, h2);
                if (b2 != b0)
                    pts[ptCount++] = interp(p2, p0, h2, h0);

                if (ptCount == 2)
                {
                    auto &layer = layerMap[k];
                    layer.height = isoHeight;
                    layer.segments.push_back({pts[0], pts[1]});
                }
            }
        }

        // 转成 vector，按高度排序
        std::vector<ContourLayer> result;
        result.reserve(layerMap.size());

        for (auto &[_, layer] : layerMap)
            result.push_back(std::move(layer));

        std::sort(result.begin(), result.end(),
                  [](const auto &a, const auto &b)
                  {
                      return a.height < b.height;
                  });

        return result;
    }

    void Terrain::setContoursShow(bool contourShow)
    {
        isShowContours = contourShow;
    }

    void Terrain::drawContours(std::vector<ContourLayer> layers) const
    {
        if (!contourShowData.isShowContours)
            return;

        // std::vector<std::vector<Vector3>> polyline_pts;

        for (ContourLayer &layer : layers)
        {
            std::vector<geo::Segment> segments = layer.segments;
            if (contourShowData.isShowLines)
            {
                for (auto &seg : segments)
                {
                    Vector3 p0 = {seg.p0.x(), seg.p0.z(), -seg.p0.y()};
                    Vector3 p1 = {seg.p1.x(), seg.p1.z(), -seg.p1.y()};
                    DrawLine3D(p0, p1, contourShowData.lineColor);
                    if (contourShowData.isShowPts)
                    {
                        DrawCube(p0, contourShowData.ptSize, contourShowData.ptSize, contourShowData.ptSize, contourShowData.ptColor);
                        DrawCube(p1, contourShowData.ptSize, contourShowData.ptSize, contourShowData.ptSize, contourShowData.ptColor);
                    }
                }
            }

            // if (contourShowData.isShowPts)
            // {
            //     std::vector<geo::Polyline> polylines = buildPolylines(segments);
            //     for (int i = 0; i < polylines.size(); ++i)
            //     {
            //         std::vector<Eigen::Vector3f> pts = polylines[i].points;
            //         if (contourShowData.isShowIndex)
            //         {
            //             for (auto &pt : pts)
            //             {
            //                 DrawCubeV({pt.x(), pt.z(), -pt.y()}, {
            //                                                          contourShowData.ptSize,
            //                                                          contourShowData.ptSize,
            //                                                          contourShowData.ptSize,
            //                                                      },
            //                           contourShowData.ptColor);
            //             }
            //             // render.draw_index_fonts(world_pts, contourShowData.ptIndexSize, contourShowData.ptIndexColor);
            //         }
            //     }
            // }
        }
    }

    void Terrain::drawContourPtIndices(std::vector<ContourLayer> layers, render::Renderer3D &render) const
    {
        if (!contourShowData.isShowContours || !contourShowData.isShowPts || !contourShowData.isShowIndex)
            return;
        for (ContourLayer &layer : layers)
        {
            std::vector<geo::Segment> segments = layer.segments;
            std::vector<geo::Polyline> polylines = buildPolylines(segments);

            for (int i = 0; i < polylines.size(); ++i)
            {
                std::vector<Eigen::Vector3f> pts = polylines[i].points;
                std::vector<Vector3> world_pts = render::vec3_to_Vector3_arr(pts);

                render.draw_index_fonts(world_pts, contourShowData.ptIndexSize, contourShowData.ptIndexColor);
            }
        }
    }

    void Terrain::buildContourSettings()
    {
        if (ImGui::Checkbox("ContoursDrawSetting", &contourShowData.isShowContours))
        {
            setContoursShow(contourShowData.isShowContours);
        }
        if (contourShowData.isShowContours)
        {
            ImGui::Indent();

            // ===== Lines =====
            if (ImGui::Checkbox("Show Lines", &contourShowData.isShowLines))
            {
                // nothing else needed
            }

            if (contourShowData.isShowLines)
            {
                ImGui::Indent();

                // Color picker（轮盘）
                if (ImGui::ColorEdit4(
                        "Line Color",
                        (float *)&contourShowData.lineColorF,
                        ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                {
                    contourShowData.lineColor = transformFloat4ToColor(contourShowData.lineColorF);
                }

                ImGui::Unindent();
            }

            ImGui::Separator();

            // ===== Points =====
            if (ImGui::Checkbox("Show Points", &contourShowData.isShowPts))
            {
            }

            if (contourShowData.isShowPts)
            {
                ImGui::Indent();

                if (ImGui::ColorEdit4(
                        "Point Color",
                        (float *)&contourShowData.ptColorF,
                        ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                {
                    contourShowData.ptColor = transformFloat4ToColor(contourShowData.ptColorF);
                }

                ImGui::SliderFloat(
                    "Point Size",
                    &contourShowData.ptSize,
                    0.02f, 0.5f,
                    "%.2f");

                ImGui::Unindent();
            }

            ImGui::Separator();

            // ===== Indices =====
            if (ImGui::Checkbox("Show Indices", &contourShowData.isShowIndex))
            {
            }

            if (contourShowData.isShowIndex)
            {
                ImGui::Indent();

                if (ImGui::ColorEdit4(
                        "Index Color",
                        (float *)&contourShowData.ptIndexColorF,
                        ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                {
                    contourShowData.ptIndexColor = transformFloat4ToColor(contourShowData.ptIndexColorF);
                }

                ImGui::SliderFloat(
                    "Index Size",
                    &contourShowData.ptIndexSize,
                    8.0f, 48.0f,
                    "%.0f");

                ImGui::Unindent();
            }

            ImGui::Unindent();
        }
    }

    Mesh buildRaylibMesh(const TerrainMeshData &src)
    {
        Mesh mesh = {0};

        mesh.vertexCount = (int)src.vertices.size();
        mesh.triangleCount = (int)src.indices.size() / 3;

        mesh.vertices = (float *)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
        mesh.normals = (float *)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
        mesh.indices = (unsigned short *)MemAlloc(mesh.triangleCount * 3 * sizeof(unsigned short));
        mesh.colors = (unsigned char *)MemAlloc(mesh.vertexCount * 4 * sizeof(unsigned char));

        // 顶点 + 法线
        for (int i = 0; i < mesh.vertexCount; ++i)
        {
            Vector3 p = render::vec3_to_Vector3(src.vertices[i].position);
            Vector3 n = render::vec3_to_Vector3(src.vertices[i].normal);

            mesh.vertices[i * 3 + 0] = p.x;
            mesh.vertices[i * 3 + 1] = p.y;
            mesh.vertices[i * 3 + 2] = p.z;

            mesh.normals[i * 3 + 0] = n.x;
            mesh.normals[i * 3 + 1] = n.y;
            mesh.normals[i * 3 + 2] = n.z;

            mesh.colors[i * 4 + 0] = 255;
            mesh.colors[i * 4 + 1] = 255;
            mesh.colors[i * 4 + 2] = 255;
            mesh.colors[i * 4 + 3] = 255;
        }

        // 反转绕序（镜像）
        for (size_t i = 0; i < src.indices.size(); i += 3)
        {
            mesh.indices[i + 0] = (unsigned short)src.indices[i + 0];
            mesh.indices[i + 1] = (unsigned short)src.indices[i + 1];
            mesh.indices[i + 2] = (unsigned short)src.indices[i + 2];
        }

        UploadMesh(&mesh, true);
        return mesh;
    }

    //------------------------------------road path finding -------------------------------------------
    void Terrain::linkEdge(int a, int b, int rank, std::vector<geo::GraphEdge> &edges) const
    {
        int maxIndex = (width + 1) * (height + 1) - 1;
        if (b < 0 || b > maxIndex)
            return;
        // no directional graph
        if (a < b)
            return;
        int ax = a % (width + 1);
        int bx = b % (width + 1);

        if (ax < rank && bx > width - rank)
            return;
        if (bx < rank && ax > width - rank)
            return;

        edges.push_back({a, b, 0.f});
        edges.push_back({b, a, 0.f});
    }

    std::vector<geo::GraphEdge> Terrain::buildGraph(int rank) const
    {
        std::vector<geo::GraphEdge> edges;

        int vertsPerRow = width + 1;
        int totalVerts = vertsPerRow * (height + 1);

        for (int v = 0; v < totalVerts; ++v)
        {
            int x = v % vertsPerRow;
            int y = v / vertsPerRow;
            if (!valid(x, y))
                continue;

            // === 已使用方向（最简整数方向）===
            std::set<std::pair<int, int>> usedDirs;

            auto tryAddEdge = [&](int dx, int dy)
            {
                int g = igcd(std::abs(dx), std::abs(dy));
                int ndx = dx / g;
                int ndy = dy / g;

                // 同方向已经连过
                if (usedDirs.count({ndx, ndy}))
                    return;

                usedDirs.insert({ndx, ndy});

                int nx = x + dx;
                int ny = y + dy;
                if (!valid(nx, ny))
                    return;

                int u = ny * vertsPerRow + nx;

                // 无向图：只建一次
                if (v < u)
                    return;

                edges.push_back({v, u, 0.f, ndx, ndy});
                edges.push_back({u, v, 0.f, -ndx, -ndy});
            };

            // ---------- rank = 0 ----------
            if (rank >= 0)
            {
                tryAddEdge(1, 0);
                tryAddEdge(-1, 0);
                tryAddEdge(0, 1);
                tryAddEdge(0, -1);
            }

            // ---------- rank = 1 ----------
            if (rank >= 1)
            {
                tryAddEdge(1, 1);
                tryAddEdge(-1, 1);
                tryAddEdge(1, -1);
                tryAddEdge(-1, -1);
            }

            // ---------- rank >= 2 ----------
            for (int r = 2; r <= rank; ++r)
            {
                // 外围正方形一圈
                for (int dy = -r; dy <= r; ++dy)
                {
                    for (int dx = -r; dx <= r; ++dx)
                    {
                        // 只取外围
                        if (std::abs(dx) != r && std::abs(dy) != r)
                            continue;

                        if (dx == 0 && dy == 0)
                            continue;

                        tryAddEdge(dx, dy);
                    }
                }
            }
        }

        return edges;
    }


    void Terrain::drawGraphEdges(int cx,int cy,int rank) const
    {
        if (!valid(cx, cy))
            return;

        Vector3 p0 = render::vec3_to_Vector3(mesh.vertices[vertexIndex(cx, cy)].position);

        std::set<std::pair<int, int>> usedDirs;

        auto drawDir = [&](int dx, int dy)
        {
            int g = igcd(std::abs(dx), std::abs(dy));
            int ndx = dx / g;
            int ndy = dy / g;

            if (usedDirs.count({ndx, ndy}))
                return;

            usedDirs.insert({ndx, ndy});

            int nx = cx + dx;
            int ny = cy + dy;
            if (!valid(nx, ny))
                return;

            Vector3 p1 = render::vec3_to_Vector3(mesh.vertices[vertexIndex(nx, ny)].position);

            Color col = ORANGE;
            if (ndx == 0 || ndy == 0)
                col = WHITE; // 轴向
            else if (std::abs(ndx) == std::abs(ndy))
                col = GREEN; // 对角线

            DrawLine3D(p0, p1, col);
        };

        // ---------- rank = 0 ----------
        if (rank >= 0)
        {
            drawDir(1, 0);
            drawDir(-1, 0);
            drawDir(0, 1);
            drawDir(0, -1);
        }

        // ---------- rank = 1 ----------
        if (rank >= 1)
        {
            drawDir(1, 1);
            drawDir(-1, 1);
            drawDir(1, -1);
            drawDir(-1, -1);
        }

        // ---------- rank >= 2 ----------
        for (int r = 2; r <= rank; ++r)
        {
            for (int dy = -r; dy <= r; ++dy)
            {
                for (int dx = -r; dx <= r; ++dx)
                {
                    if (std::abs(dx) != r && std::abs(dy) != r)
                        continue;

                    if (dx == 0 && dy == 0)
                        continue;

                    drawDir(dx, dy);
                }
            }
        }
    }

    void Terrain::debugBuildGraphAt(int cx, int cy, int rank) const
    {
        std::cout << "=== Debug Graph at (" << cx << "," << cy
                  << "), rank = " << rank << " ===\n";

        // 已使用方向集合（方向 = 最简整数比）
        std::set<std::pair<int, int>> usedDirs;

        auto tryAdd = [&](int dx, int dy)
        {
            int g = igcd(std::abs(dx), std::abs(dy));
            dx /= g;
            dy /= g;

            if (usedDirs.count({dx, dy}))
                return;

            usedDirs.insert({dx, dy});

            int nx = cx + dx * g;
            int ny = cy + dy * g;

            if (!valid(nx, ny))
                return;

            std::cout << "  dir=(" << dx << "," << dy
                      << ")  -> (" << nx << "," << ny << ")\n";
        };

        // ---------- rank = 0 ----------
        if (rank >= 0)
        {
            tryAdd(1, 0);
            tryAdd(-1, 0);
            tryAdd(0, 1);
            tryAdd(0, -1);
        }

        // ---------- rank = 1 ----------
        if (rank >= 1)
        {
            tryAdd(1, 1);
            tryAdd(-1, 1);
            tryAdd(1, -1);
            tryAdd(-1, -1);
        }

        // ---------- rank >= 2 ----------
        for (int r = 2; r <= rank; ++r)
        {
            std::cout << "-- rank " << r
                      << " square size = " << (2 * r) << " --\n";

            for (int dy = -r; dy <= r; ++dy)
            {
                for (int dx = -r; dx <= r; ++dx)
                {
                    // 只取外围
                    if (std::abs(dx) != r && std::abs(dy) != r)
                        continue;

                    if (dx == 0 && dy == 0)
                        continue;

                    tryAdd(dx, dy);
                }
            }
        }
    }

    float Terrain::edgeCost(int a, int b) const
    {
        const Eigen::Vector3f &p0 = mesh.vertices[a].position;
        const Eigen::Vector3f &p1 = mesh.vertices[b].position;

        // ---- 3D 实际距离（这是 dist）----
        float dist3D = (p1 - p0).norm();

        // ---- 水平距离（只用于算坡度）----
        float horiz = (p1.head<2>() - p0.head<2>()).norm();

        // ---- 高差 ----
        float dh = p1.z() - p0.z();
        float absDh = std::abs(dh);

        // ---- 坡度（rise / run）----
        float slope = absDh / (horiz + 1e-4f);

        // ---- 上下坡惩罚 ----
        float verticalPenalty = 0.0f;
        if (dh > 0)
            verticalPenalty = dh * w_up; // 上坡
        else
            verticalPenalty = (-dh) * w_down; // 下坡

        // ---- 综合 cost ----
        float cost =
            dist3D * w_dist + // 距离
            slope * w_slope + // 坡度惩罚
            verticalPenalty;  // 上/下坡额外惩罚

        return cost;
    }

    std::vector<int> Terrain::shortestPath(
        int start,
        int goal,
        const std::vector<std::vector<int>> &adj) const
    {
        const int N = adj.size();
        const float INF = std::numeric_limits<float>::infinity();

        std::vector<float> dist(N, INF);
        std::vector<int> prev(N, -1);

        using Node = std::pair<float, int>; // (cost, vertex)
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;

        dist[start] = 0.0f;
        pq.push({0.0f, start});

        while (!pq.empty())
        {
            auto [curDist, u] = pq.top();
            pq.pop();

            if (u == goal)
                break;

            if (curDist > dist[u])
                continue;

            for (int v : adj[u])
            {
                float cost = edgeCost(u, v);
                float nd = curDist + cost;

                if (nd < dist[v])
                {
                    dist[v] = nd;
                    prev[v] = u;
                    pq.push({nd, v});
                }
            }
        }

        // ---- 回溯路径 ----
        std::vector<int> path;
        if (prev[goal] == -1)
            return path; // 无路

        for (int v = goal; v != -1; v = prev[v])
            path.push_back(v);

        std::reverse(path.begin(), path.end());
        return path;
    }

    std::vector<std::vector<geo::GraphEdge>> Terrain::buildAdjacencyGraph(int rank) const
    {
        int vertsPerRow = width + 1;
        int totalVerts = vertsPerRow * (height + 1);

        std::vector<std::vector<geo::GraphEdge>> adj(totalVerts);

        auto edges = buildGraph(rank);

        for (const auto &e : edges)
        {
            // 计算真实 cost（统一在这里）
            float cost = edgeCost(e.from, e.to);

            geo::GraphEdge edge = e;
            edge.cost = cost;

            adj[e.from].push_back(edge);
        }

        return adj;
    }

    std::vector<int> Terrain::shortestPathDijkstra(
        int start,
        int goal,
        const std::vector<std::vector<geo::GraphEdge>> &adj) const
    {
        const float INF = std::numeric_limits<float>::infinity();
        int n = adj.size();

        std::vector<float> dist(n, INF);
        std::vector<int> prev(n, -1);

        using Node = std::pair<float, int>; // (dist, vertex)
        std::priority_queue<Node, std::vector<Node>, std::greater<>> pq;

        dist[start] = 0.0f;
        pq.push({0.0f, start});

        while (!pq.empty())
        {
            auto [d, u] = pq.top();
            pq.pop();

            if (d > dist[u])
                continue;

            if (u == goal)
                break;

            for (const auto &e : adj[u])
            {
                int v = e.to;
                float nd = d + e.cost;

                if (nd < dist[v])
                {
                    dist[v] = nd;
                    prev[v] = u;
                    pq.push({nd, v});
                }
            }
        }

        // ---- 回溯路径 ----
        std::vector<int> path;
        if (prev[goal] == -1)
            return path; // no path

        for (int v = goal; v != -1; v = prev[v])
            path.push_back(v);

        std::reverse(path.begin(), path.end());
        return path;
    }

    void Terrain::drawPath(const std::vector<int> &path,float width) const
    {
        for (int i = 1; i < path.size(); ++i)
        {
            const auto &p0 = mesh.vertices[path[i - 1]].position;
            const auto &p1 = mesh.vertices[path[i]].position;
            DrawCylinderEx(render::vec3_to_Vector3(p0), render::vec3_to_Vector3(p1),width,width,1,RED);
        }
    }
}