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

    Terrain::Terrain(int seed_, int w, int h, float cs, float frequency, float amplitude)
        : seed(seed_), width(w), height(h), cellSize(cs)
    {
        heightmap.resize((width + 1) * (height + 1), 0.f);
        generateHeight(frequency, amplitude);
        buildMesh();
        calculateInfos();
        upload();
    }

    void Terrain::generateHeight(float freq, float amp)
    {
        Noise noise(seed);
        for (int y = 0; y <= height; y++)
        {
            for (int x = 0; x <= width; x++)
            {
                float nx = (x + 32.34f) * freq;
                float ny = (y + 212.3f) * freq;

                heightmap[y * (width + 1) + x] = fbm2D_OpenSimplex(noise, nx, ny, octaves, lacunarity, gain) * amp;
            }
        }
    }

    void Terrain::upload()
    {
        chunkMeshes = buildRaylibMeshes(mesh, 128);

        models.resize(chunkMeshes.size());

        for (int i = 0; i < chunkMeshes.size(); ++i)
        {
            models[i] = LoadModelFromMesh(chunkMeshes[i].mesh);
        }

        // std::cout << "mesh build all" << std::endl;
    }

    void Terrain::regenerate(int w, int h, float freq, float amp)
    {
        width = w;
        height = h;

        heightmap.clear();
        mesh.vertices.clear();
        mesh.indices.clear();

        heightmap.resize((width + 1) * (height + 1), 0.f);

        generateHeight(freq, amp);
        buildMesh();
        calculateInfos();
        // 释放旧 GPU 资源
        for (Model &model : models)
            UnloadModel(model);

        upload();
    }

    void Terrain::regenerate(int seed_, int w, int h, float freq, float amp)
    {
        seed = seed_;
        width = w;
        height = h;

        heightmap.clear();
        mesh.vertices.clear();
        mesh.indices.clear();

        heightmap.resize((width + 1) * (height + 1), 0.f);

        generateHeight(freq, amp);
        buildMesh();
        calculateInfos();

        // 释放旧 GPU 资源
        for (Model &model : models)
            UnloadModel(model);

        upload();
    }

    void Terrain::draw() const
    {

        switch (viewMode)
        {
        case TerrainViewMode::Wire:
            for (const Model &model : models)
                DrawModelWires(model, {0, 0, 0}, 1.0f, BLACK);
            break;

        case TerrainViewMode::Aspect:
        case TerrainViewMode::Slope:
            for (const Model &model : models)
                DrawModel(model, {0, 0, 0}, 1.0f, WHITE);
            break;

        case TerrainViewMode::Lit:
        default:
            for (const Model &model : models)
                DrawModel(model, {0, 0, 0}, 1.0f, GRAY);
            break;
        }
    }

    // void Terrain::applyFaceColor()
    // {
    //     Mesh &m = model.meshes[0];

    //     // 1. 确保有 colors buffer
    //     if (!m.colors)
    //     {
    //         m.colors = (unsigned char *)MemAlloc(m.vertexCount * 4);
    //     }

    //     // 2. 默认底色
    //     for (int i = 0; i < m.vertexCount; ++i)
    //     {
    //         m.colors[i * 4 + 0] = 200;
    //         m.colors[i * 4 + 1] = 200;
    //         m.colors[i * 4 + 2] = 200;
    //         m.colors[i * 4 + 3] = 255;
    //     }
    //     std::vector<float> scores = evaluateFaceScore();

    //     std::vector<std::vector<int>> regions = floodFillFaces(scores, score_threshold);
    //     size_t faceCount = faceInfos.size();
    //     std::vector<int> faceRegionId(faceCount, -1);
    //     std::vector<Color> regionColors;

    //     int validRegionIndex = 0;
    //     for (const auto &region : regions)
    //     {
    //         if ((int)region.size() < minRegionFaceSize)
    //             continue;

    //         float hue = std::fmod(validRegionIndex * 0.6180339f, 1.0f); // 黄金分割
    //         Color rc = renderUtil::ColorFromHue(hue);

    //         for (int f : region)
    //             faceRegionId[f] = validRegionIndex;

    //         regionColors.push_back(rc);
    //         validRegionIndex++;
    //     }
    //     // 3. 按 face 着色
    //     for (size_t f = 0; f < faceInfos.size(); ++f)
    //     {
    //         const auto &face = faceInfos[f];

    //         Color c{180, 180, 180, 255};

    //         // 平地判断
    //         bool isFlat = face.slope < 1e-4f;
    //         int rid = faceRegionId[f];

    //         if (!isFlat)
    //         {
    //             if (viewMode == TerrainViewMode::Aspect)
    //             {
    //                 // aspect ∈ [0, 2π)
    //                 c = renderUtil::AspectToColor(face.aspect);
    //             }
    //             else if (viewMode == TerrainViewMode::Slope)
    //             {
    //                 // slope ∈ [0, π/2]
    //                 float t = face.slope / (PI * 0.5f);
    //                 t = std::clamp(t, 0.0f, 1.0f);

    //                 // 蓝 → 红（缓坡 → 陡坡）
    //                 c = {
    //                     (unsigned char)(255 * t),
    //                     0,
    //                     (unsigned char)(255 * (1.0f - t)),
    //                     255};
    //             }
    //             else if (viewMode == TerrainViewMode::Score)
    //             {
    //                 if (rid >= 0)
    //                     c = regionColors[rid];
    //                 else
    //                 {
    //                     //==== 2. fallback：原 score 渐变 ====
    //                     const float &score = scores[f];

    //                     if (score >= 0)
    //                     {
    //                         float gamma = 1.0f / 1.5f;
    //                         float s = std::pow(score, gamma);
    //                         c = {
    //                             (unsigned char)(255 * s),
    //                             (unsigned char)(255 * s),
    //                             (unsigned char)(255 * (1.0f - s)),
    //                             255};

    //                         // float hue = (1.0f - score) * 220.0f + score * 60.0f;
    //                         // c = ColorFromHSV(
    //                         //     hue,
    //                         //     0.85f, // saturation
    //                         //     0.95f  // value
    //                         // );
    //                     }
    //                     else
    //                     {
    //                         // 错误 / 不可用区域
    //                         c = {255, 0, 0, 255};
    //                     }
    //                 }
    //             }
    //         }

    //         int i0 = mesh.indices[f * 3 + 0];
    //         int i1 = mesh.indices[f * 3 + 1];
    //         int i2 = mesh.indices[f * 3 + 2];

    //         auto setColor = [&](int vi)
    //         {
    //             m.colors[vi * 4 + 0] = c.r;
    //             m.colors[vi * 4 + 1] = c.g;
    //             m.colors[vi * 4 + 2] = c.b;
    //             m.colors[vi * 4 + 3] = 255;
    //         };

    //         setColor(i0);
    //         setColor(i1);
    //         setColor(i2);
    //     }

    //     // 4. 上传到 GPU
    //     UpdateMeshBuffer(m, 3, m.colors, m.vertexCount * 4, 0);
    // }

    void Terrain::applyFaceColor()
    {
        std::vector<int> faceRegionId(faceInfos.size(), -1);
        std::vector<Color> regionColors;

        int rid = 0;
        for (auto &r : regions)
        {
            if ((int)r.size() < minRegionFaceSize)
                continue;

            Color c = renderUtil::ColorFromHue(
                std::fmod(rid * 0.6180339f, 1.0f));

            for (int f : r)
                faceRegionId[f] = rid;
            regionColors.push_back(c);
            rid++;
        }

        // std::cout <<"apply begin" <<std::endl;
        //  ==== 遍历每个 chunk ====
        for (auto &chunk : chunkMeshes)
        {
            Mesh &m = chunk.mesh;

            // 1. 先清底色
            for (int i = 0; i < m.vertexCount; ++i)
            {
                m.colors[i * 4 + 0] = 200;
                m.colors[i * 4 + 1] = 200;
                m.colors[i * 4 + 2] = 200;
                m.colors[i * 4 + 3] = 255;
            }

            // 2. 按 local face 着色
            for (int lf = 0; lf < (int)chunk.localToGlobalFace.size(); ++lf)
            {
                int gf = chunk.localToGlobalFace[lf];
                const auto &face = faceInfos[gf];

                Color c{180, 180, 180, 255};
                int rid = faceRegionId[gf];

                if (viewMode == TerrainViewMode::Aspect)
                {
                    c = renderUtil::AspectToColor(face.aspect);
                }
                else if (viewMode == TerrainViewMode::Slope)
                {
                    float t = face.slope / (PI * 0.5f);
                    t = std::clamp(t, 0.f, 1.f);
                    c = {
                        (unsigned char)(255 * t),
                        0,
                        (unsigned char)(255 * (1.f - t)),
                        255};
                }
                else if (viewMode == TerrainViewMode::Score)
                {
                    if (rid >= 0)
                        c = regionColors[rid];
                    else
                    {
                        //==== 2. fallback：原 score 渐变 ====
                        const float &score = scores[gf];

                        if (score >= 0)
                        {
                            float gamma = 1.0f / 1.5f;
                            float s = std::pow(score, gamma);
                            c = {
                                (unsigned char)(255 * s),
                                (unsigned char)(255 * s),
                                (unsigned char)(255 * (1.0f - s)),
                                255};

                            // float hue = (1.0f - score) * 220.0f + score * 60.0f;
                            // c = ColorFromHSV(
                            //     hue,
                            //     0.85f, // saturation
                            //     0.95f  // value
                            // );
                        }
                        else
                        {
                            // 错误 / 不可用区域
                            c = {255, 0, 0, 255};
                        }
                    }
                }

                // local face -> local indices
                int li0 = m.indices[lf * 3 + 0];
                int li1 = m.indices[lf * 3 + 1];
                int li2 = m.indices[lf * 3 + 2];

                auto setColor = [&](int lv)
                {
                    m.colors[lv * 4 + 0] = c.r;
                    m.colors[lv * 4 + 1] = c.g;
                    m.colors[lv * 4 + 2] = c.b;
                    m.colors[lv * 4 + 3] = 255;
                };

                setColor(li0);
                setColor(li1);
                setColor(li2);
            }
            std::cout << std::endl;

            UpdateMeshBuffer(m, 3, m.colors, m.vertexCount * 4, 0);
        }
    }

    void Terrain::setViewMode(TerrainViewMode mode)
    {
        viewMode = mode;

        if (viewMode != TerrainViewMode::Wire)
        {
            applyFaceColor();
        }
    }

    const std::vector<Vector3> Terrain::getMeshVertices()
    {
        std::vector<Vector3> vertices;
        vertices.reserve(mesh.vertices.size());
        for (auto &v : mesh.vertices)
        {
            vertices.push_back({v.position.x(), v.position.z(), -v.position.y()});
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
            Eigen::Vector2f h(n.x(), n.y());

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

            // 坡向（使用下坡方向）
            Eigen::Vector3f down = -v.normal;
            Eigen::Vector2f h(v.normal.x(), v.normal.y());

            float aspect = 0.f;
            if (h.norm() > 1e-5f)
            {
                aspect = atan2(h.y(), h.x());
                if (aspect < 0)
                    aspect += 2.0f * PI;
            }

            float slope = acos(std::clamp(v.normal.z(), -1.0f, 1.0f));
            v.aspect = aspect;
            v.slope = slope;
        }
    }

    void Terrain::calculateInfos()
    {
        scores = evaluateFaceScore();
        vertexScores = evaluateVertexScore(scores);
        regions = floodFillFaces(scores, score_threshold);
        regionInfos.clear();
        for (const auto &region : regions)
        {
            if (region.size() > minRegionFaceSize)
            {
                RegionInfo regionInfo{region};
                regionInfos.push_back(regionInfo);
            }
        }

        for (RegionInfo &region : regionInfos)
            computeRegionCenter(region);

        // for(int i = 0; i < regions.size(); i++){
        //     std::cout << "cur " << i << " region num is " << regions[i].size() <<std::endl;
        // }
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

    void Terrain::drawContours(const std::vector<ContourLayer> &layers) const
    {
        if (!contourShowData.isShowContours)
            return;

        // std::vector<std::vector<Vector3>> polyline_pts;

        for (const ContourLayer &layer : layers)
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

    void Terrain::drawContourPtIndices(const std::vector<ContourLayer> &layers, render::Renderer3D &render) const
    {
        if (!contourShowData.isShowContours || !contourShowData.isShowPts || !contourShowData.isShowIndex)
            return;
        for (const ContourLayer &layer : layers)
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

        for (size_t i = 0; i < src.indices.size(); i += 3)
        {
            mesh.indices[i + 0] = (unsigned short)src.indices[i + 0];
            mesh.indices[i + 1] = (unsigned short)src.indices[i + 1];
            mesh.indices[i + 2] = (unsigned short)src.indices[i + 2];
        }

        UploadMesh(&mesh, true);
        return mesh;
    }

    std::vector<ChunkMeshInfo> buildRaylibMeshes(const TerrainMeshData &src, int chunkSize)
    {
        std::vector<ChunkMeshInfo> chunks;

        const int width = src.gridWidth;
        const int height = src.gridHeight;
        const int facesPerRow = width * 2;

        for (int cy = 0; cy < height; cy += chunkSize)
        {
            for (int cx = 0; cx < width; cx += chunkSize)
            {
                std::unordered_map<int, int> g2l;
                std::vector<int> l2gv;
                std::vector<int> localToGlobalFace;
                std::vector<unsigned short> localIndices;

                int faceStart = cy * facesPerRow + cx * 2;

                int faceW = std::min(chunkSize, width - cx) * 2;
                int faceH = std::min(chunkSize, height - cy);

                for (int fy = 0; fy < faceH; ++fy)
                {
                    int row = faceStart + fy * facesPerRow;

                    for (int fx = 0; fx < faceW; ++fx)
                    {
                        int face = row + fx;
                        int lf = (int)localToGlobalFace.size();
                        localToGlobalFace.push_back(face);

                        for (int k = 0; k < 3; ++k)
                        {
                            int gvi = src.indices[face * 3 + k];

                            auto it = g2l.find(gvi);
                            if (it == g2l.end())
                            {
                                int lvi = (int)l2gv.size();
                                g2l[gvi] = lvi;
                                l2gv.push_back(gvi);
                                localIndices.push_back((unsigned short)lvi);
                            }
                            else
                            {
                                localIndices.push_back((unsigned short)it->second);
                            }
                        }
                    }
                }

                // ==== build raylib Mesh ====

                Mesh m = {0};
                m.vertexCount = (int)l2gv.size();
                m.triangleCount = (int)localIndices.size() / 3;

                m.vertices = (float *)MemAlloc(m.vertexCount * 3 * sizeof(float));
                m.normals = (float *)MemAlloc(m.vertexCount * 3 * sizeof(float));
                m.colors = (unsigned char *)MemAlloc(m.vertexCount * 4);
                m.indices = (unsigned short *)MemAlloc(m.triangleCount * 3 * sizeof(unsigned short));

                for (int i = 0; i < m.vertexCount; ++i)
                {
                    const auto &v = src.vertices[l2gv[i]];

                    m.vertices[i * 3 + 0] = v.position.x();
                    m.vertices[i * 3 + 1] = v.position.z();
                    m.vertices[i * 3 + 2] = -v.position.y();

                    m.normals[i * 3 + 0] = v.normal.x();
                    m.normals[i * 3 + 1] = v.normal.z();
                    m.normals[i * 3 + 2] = -v.normal.y();

                    m.colors[i * 4 + 0] = 200;
                    m.colors[i * 4 + 1] = 200;
                    m.colors[i * 4 + 2] = 200;
                    m.colors[i * 4 + 3] = 255;
                }

                memcpy(m.indices, localIndices.data(), localIndices.size() * sizeof(unsigned short));

                UploadMesh(&m, true);

                chunks.push_back({m, l2gv, localToGlobalFace});
            }
        }
        // std::cout << "chunks build succeed with " << chunks.size() << std::endl;

        return chunks;
    }

    //------------------------------------road path finding -------------------------------------------

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

    void Terrain::drawGraphEdges(int cx, int cy, int rank) const
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

    std::vector<std::vector<geo::GraphEdge>> Terrain::buildAdjacencyGraph(int rank) const
    {
        int vertsPerRow = width + 1;
        int totalVerts = vertsPerRow * (height + 1);

        std::vector<std::vector<geo::GraphEdge>> adj(totalVerts);

        auto edges = buildGraph(rank);

        for (const auto &e : edges)
        {
            // 计算真实 cost
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
        size_t n = adj.size();

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

    void Terrain::drawPath(const std::vector<int> &path, float width) const
    {
        if (path.size() < 2)
            return;
        for (int i = 1; i < path.size(); ++i)
        {
            const auto &p0 = mesh.vertices[path[i - 1]].position;
            const auto &p1 = mesh.vertices[path[i]].position;
            DrawCylinderEx(render::vec3_to_Vector3(p0), render::vec3_to_Vector3(p1), width, width, 1, RED);
        }
    }

    void Terrain::computeRegionCenter(RegionInfo &region) const
    {
        Eigen::Vector3f avg(0, 0, 0);
        // 1️. 计算 region 所有 face 中心的平均
        for (int f : region.faces)
            avg += faceCenter3D(f);

        avg *= (1.0f / region.faces.size());
        // 2️.找最接近 avg 的 face
        float best = std::numeric_limits<float>::max();
        for (int f : region.faces)
        {
            float d = (faceCenter3D(f) - avg).squaredNorm();
            if (d < best)
            {
                best = d;
                region.centerFace = f;
            }
        }
        // 3️.用 centerFace 的第一个 vertex 作为 seed
        region.centeridx = mesh.indices[region.centerFace * 3 + 0];
    }

    Eigen::Vector3f Terrain::computeForwardDirection(const std::vector<int> &path, int i) const
    {
        if (i == 0)
            return (mesh.vertices[path[1]].position -
                    mesh.vertices[path[0]].position)
                .normalized();

        if (i == (int)path.size() - 1)
            return (mesh.vertices[path[i]].position -
                    mesh.vertices[path[i - 1]].position)
                .normalized();

        Eigen::Vector3f a = mesh.vertices[path[i]].position -
                            mesh.vertices[path[i - 1]].position;
        Eigen::Vector3f b = mesh.vertices[path[i + 1]].position -
                            mesh.vertices[path[i]].position;

        return (a + b).normalized();
    }

    std::vector<int> Terrain::sampleVerticesByDistance(
        const std::vector<int> &path,
        float interval) const
    {
        std::vector<int> result;
        float acc = 0.0f;

        for (size_t i = 1; i < path.size(); ++i)
        {
            float d =
                (mesh.vertices[path[i]].position -
                 mesh.vertices[path[i - 1]].position)
                    .norm();

            acc += d;

            if (acc >= interval)
            {
                result.push_back(path[i]);
                acc = 0.0f;
            }
        }

        return result;
    }

    float Terrain::pathLength(const std::vector<int> &path) const
    {
        float len = 0.0f;
        for (size_t i = 1; i < path.size(); ++i)
        {
            len += (mesh.vertices[path[i]].position -
                    mesh.vertices[path[i - 1]].position)
                       .norm();
        }
        return len;
    }

 

 

  
 

   

  
    std::vector<std::vector<int>> Terrain::buildMainRoads(
        std::vector<RegionInfo> &regions,
        int mainRegionCount,
        const std::vector<std::vector<geo::GraphEdge>> &adj) const
    {
        std::vector<std::vector<int>> roads;

        if (regions.empty())
            return roads;

        // 1️ 按 region size 排序（降序）
        std::sort(regions.begin(), regions.end(), [](const RegionInfo &a, const RegionInfo &b)
                  { return a.faces.size() > b.faces.size(); });

        // for (RegionInfo &region : regions)
        //     computeRegionCenter(region);

        int n = std::min((int)regions.size(), mainRegionCount);

        // 2️ 少于 2 个，直接返回
        if (n < 2)
            return roads;

        // 3️ 恰好 2 个，直接连
        if (n == 2)
        {
            roads.push_back(
                shortestPathDijkstra(
                    regions[0].centeridx,
                    regions[1].centeridx,
                    adj));
            return roads;
        }

        // 4️ n >= 3：EMST (Prim)
        std::vector<bool> used(n, false);
        std::vector<float> minCost(n, std::numeric_limits<float>::max());
        std::vector<int> parent(n, -1);

        auto dist = [&](int i, int j)
        {
            const auto &p = mesh.vertices[regions[i].centeridx].position;
            const auto &q = mesh.vertices[regions[j].centeridx].position;
            return (p - q).squaredNorm();
        };

        minCost[0] = 0.0f;

        for (int it = 0; it < n; ++it)
        {
            int u = -1;
            float best = std::numeric_limits<float>::max();

            for (int i = 0; i < n; ++i)
            {
                if (!used[i] && minCost[i] < best)
                {
                    best = minCost[i];
                    u = i;
                }
            }

            if (u == -1)
                break;

            used[u] = true;

            if (parent[u] != -1)
            {
                roads.push_back(
                    shortestPathDijkstra(
                        regions[parent[u]].centeridx,
                        regions[u].centeridx,
                        adj));
            }

            for (int v = 0; v < n; ++v)
            {
                if (used[v])
                    continue;

                float d = dist(u, v);
                if (d < minCost[v])
                {
                    minCost[v] = d;
                    parent[v] = u;
                }
            }
        }

        return roads;
    }

    std::vector<Road> Terrain::buildRoads(
        std::vector<Eigen::Vector3f> &seedPoints,
        std::vector<Eigen::Vector3f> &controlPts,
        std::vector<RegionInfo> &regions,
        int mainRegionCount,
        const std::vector<std::vector<geo::GraphEdge>> &adj)
    {
        std::vector<Road> roads;
        std::unordered_set<int> occupied;
        seedPoints.clear();
        controlPts.clear();

        // ===== 1. 主干路 =====
        auto mainRoads = buildMainRoads(regions, mainRegionCount, adj);
        if (mainRoads.empty())
            return roads;

        // 记录主干路 & occupied
        for (auto &p : mainRoads)
        {
            std::vector<int> seedPointsIndices = sampleVerticesByDistance(p, 35.0f);
            for( int idx : seedPointsIndices)
                seedPoints.push_back(mesh.vertices[idx].position);
            roads.push_back({p, 1});
            for (int v : p){
                occupied.insert(v);
                controlPts.push_back(mesh.vertices[v].position);
            }
                
        }

        return roads;
    }

    void Terrain::drawRoads(const std::vector<Road> &roads, float MaxWidth) const{

    }

    std::vector<float> Terrain::evaluateVertexScore(const std::vector<float> &faceScores) const
    {
        std::vector<float> scores(mesh.vertices.size());
        std::vector<float> vertexAreaSum(mesh.vertices.size());
        const size_t V = mesh.vertices.size();
        const size_t F = mesh.indices.size() / 3;

        scores.assign(V, 0.0f);
        vertexAreaSum.assign(V, 0.0f);

        for (size_t f = 0; f < F; ++f)
        {
            int i0 = mesh.indices[3 * f + 0];
            int i1 = mesh.indices[3 * f + 1];
            int i2 = mesh.indices[3 * f + 2];

            const Eigen::Vector3f &v0 = mesh.vertices[i0].position;
            const Eigen::Vector3f &v1 = mesh.vertices[i1].position;
            const Eigen::Vector3f &v2 = mesh.vertices[i2].position;

            Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
            float area = n.norm(); // 和你法线代码一致

            float s = faceScores[f]; // 已归一化

            scores[i0] += s * area;
            scores[i1] += s * area;
            scores[i2] += s * area;

            vertexAreaSum[i0] += area;
            vertexAreaSum[i1] += area;
            vertexAreaSum[i2] += area;
        }

        for (size_t v = 0; v < V; ++v)
        {
            if (vertexAreaSum[v] > 1e-6f)
                scores[v] /= vertexAreaSum[v];
            else
                scores[v] = 0.0f;
        }
        return scores;
    }

    std::vector<float> Terrain::evaluateFaceScore() const
    {
        std::vector<float> scores(faceInfos.size());
        float slope_max = 1.04719f;
        // 南向角度
        float south = 3.0f * PI / 2.0f;
        float slope_factor = wv_slope / (wv_slope + wv_aspect);
        float aspect_factor = wv_aspect / (wv_slope + wv_aspect);

        for (int i = 0; i < faceInfos.size(); i++)
        {
            const auto &f = faceInfos[i];
            float slope_score = 0.f;
            float aspect_score = 0.f;
            float score = 0.f;
            // 角度差（wrap）
            float da = abs(f.aspect - south);
            da = std::min(da, 2 * PI - da);

            // 映射到 [0,1]
            aspect_score = 1 - da / PI;

            if (f.slope >= slope_max)
                score = -1;
            else
            {
                slope_score = 1 - std::clamp(f.slope / slope_max, 0.f, 1.f);
                score = slope_score * slope_factor + aspect_score * aspect_factor;
            }

            scores[i] = score;
            // std::cout<<"the "<<i<<" face score is "<< score <<" , ";
        }
        // std::cout<<std::endl;
        return scores;
    }

    std::vector<std::vector<int>> Terrain::buildFaceAdjacency() const
    {
        size_t faceCount = mesh.indices.size() / 3;
        std::vector<std::vector<int>> adj(faceCount);

        // edge -> face map
        std::unordered_map<uint64_t, int> edgeMap;

        auto makeKey = [](int a, int b)
        {
            if (a > b)
                std::swap(a, b);
            return (uint64_t(a) << 32) | uint64_t(b);
        };

        for (int f = 0; f < faceCount; ++f)
        {
            int i0 = mesh.indices[f * 3 + 0];
            int i1 = mesh.indices[f * 3 + 1];
            int i2 = mesh.indices[f * 3 + 2];

            int edges[3][2] = {{i0, i1}, {i1, i2}, {i2, i0}};

            for (auto &e : edges)
            {
                uint64_t key = makeKey(e[0], e[1]);
                auto it = edgeMap.find(key);
                if (it == edgeMap.end())
                {
                    edgeMap[key] = f;
                }
                else
                {
                    int other = it->second;
                    adj[f].push_back(other);
                    adj[other].push_back(f);
                }
            }
        }
        return adj;
    }

    std::vector<std::vector<int>> Terrain::floodFillFaces(const std::vector<float> &scores, float threshold) const
    {

        auto adj = buildFaceAdjacency();

        size_t n = faceInfos.size();
        std::vector<bool> visited(n, false);

        std::vector<std::vector<int>> regions;

        for (int f = 0; f < n; ++f)
        {
            if (visited[f])
                continue;
            if (scores[f] < threshold)
                continue;

            // 开始一个新区域
            std::vector<int> region;
            std::queue<int> q;

            visited[f] = true;
            q.push(f);

            while (!q.empty())
            {
                int cur = q.front();
                q.pop();
                region.push_back(cur);

                for (int nb : adj[cur])
                {
                    if (visited[nb])
                        continue;
                    if (scores[nb] < threshold)
                        continue;

                    visited[nb] = true;
                    q.push(nb);
                }
            }

            regions.push_back(region);
        }

        return regions;
    }

    std::vector<std::vector<int>> Terrain::floodFillFacesSoft(std::vector<float> &scores, float threshold) const
    {
        scores = evaluateFaceScore();
        auto adj = buildFaceAdjacency();

        const int n = faceInfos.size();
        std::vector<bool> visited(n, false);

        std::vector<std::vector<int>> regions;

        for (int seed = 0; seed < n; ++seed)
        {
            if (visited[seed])
                continue;
            if (scores[seed] < threshold)
                continue;

            std::vector<int> region;
            std::queue<int> q;

            int badCount = 0;

            visited[seed] = true;
            q.push(seed);

            const Eigen::Vector2f seedPos = faceCenter(seed);

            while (!q.empty())
            {
                int cur = q.front();
                q.pop();
                region.push_back(cur);

                if (scores[cur] < threshold)
                    badCount++;

                // ---- 坏点比例限制 ----
                if ((float)badCount / region.size() > regionConfig.badRatioLimit)
                    break;

                for (int nb : adj[cur])
                {
                    if (visited[nb])
                        continue;

                    // ---- 硬拒绝：特别差的 ----
                    if (scores[nb] < 0)
                        continue;

                    // ---- 半径限制（防止拉丝）----
                    // float d = (faceCenter(nb) - seedPos).norm();
                    // if (d > regionConfig.maxRadius)
                    //     continue;

                    visited[nb] = true;
                    q.push(nb);
                }
            }

            if ((int)region.size() >= minRegionFaceSize /* &&
                isRegionCompact(region, seedPos) */
            )
            {
                regions.push_back(region);
            }
        }

        return regions;
    }

    bool Terrain::isRegionCompact(const std::vector<int> &region, const Eigen::Vector2f &center) const
    {
        float maxDist = 0.f;
        float avgDist = 0.f;

        for (int f : region)
        {
            float d = (faceCenter(f) - center).norm();
            maxDist = std::max(maxDist, d);
            avgDist += d;
        }

        avgDist /= region.size();

        // 太细长
        if (maxDist > avgDist * 2.5f)
            return false;

        return true;
    }

    std::vector<int> Terrain::computeSeedRegion(int seedFace, const std::vector<float> &scores) const
    {
        Eigen::Vector2f seedPos = faceCenter(seedFace);

        // 1. 候选 face 预筛
        std::vector<int> candidates;
        for (int f = 0; f < faceInfos.size(); ++f)
        {
            if ((faceCenter(f) - seedPos).norm() <= regionConfig.radius * regionConfig.targetCount)
                candidates.push_back(f);
        }

        std::vector<int> coverCount(candidates.size(), 0);
        std::vector<Circle> circles;
        circles.push_back({seedPos, regionConfig.radius});

        std::vector<int> region;

        while (true)
        {
            // 2. 增量更新 coverCount
            const Circle &newC = circles.back();
            for (int i = 0; i < candidates.size(); ++i)
            {
                if (insideCircle(candidates[i], newC))
                    coverCount[i]++;
            }

            // 3. 统计 region
            region.clear();
            int badCount = 0;

            for (int i = 0; i < candidates.size(); ++i)
            {
                if (coverCount[i] >= 3)
                {
                    int f = candidates[i];
                    region.push_back(f);
                    if (scores[f] < 0)
                        badCount++;
                }
            }

            if ((int)region.size() >= regionConfig.targetCount)
                break;

            if (!region.empty() &&
                (float)badCount / region.size() > regionConfig.badRatioLimit)
                break;

            // 4. 找最远可接受点
            int best = -1;
            float bestDist = 0.f;

            for (int f : region)
            {
                if (scores[f] < score_threshold)
                    continue;

                float d = (faceCenter(f) - seedPos).norm();
                if (d > bestDist)
                {
                    bestDist = d;
                    best = f;
                }
            }

            if (best < 0)
                break;

            circles.push_back({faceCenter(best), regionConfig.radius});
        }

        return region;
    }
}