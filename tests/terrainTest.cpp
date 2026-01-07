#include <iostream>
#include "render.h"
#include "terrain.h"
#include "renderUtil.h"
#include <rlImGui.h>

using render::Renderer3D;
using terrain::Terrain, terrain::TerrainCell, terrain::TerrainViewMode, terrain::ContourLayer;

int main()
{
    std::cout << "Hello, TerrainTest Start!" << std::endl;
    Renderer3D render(1920, 1080, 45.0f, CAMERA_PERSPECTIVE, "TerrainTest");
    
    // === 静态变量：ImGui 需要跨帧保存 ===
    static int debugCx = 64;
    static int debugCy = 64;
    static int debugRank = 3;
    static float pathWidth = 0.05f;

    static int start = 0;
    static int target = 210;
    int lastStart = -1;
    int lastTarget = -1;
    int lastRank = -1;

    static float w_slope = 53.5f;
    static float w_dist = 10.5f;
    static float w_up = 24.5f;
    static float w_down = 14.5f;

    static float score_threshold = 0.5f;
    static int terrain_width = 128;
    static int terrainPow = 8;
    static float frequency = 0.03f;
    static float amplitude = 6.f;
    Terrain terrain((unsigned)time(nullptr) ,1 << terrainPow, 1 << terrainPow, 1.f, frequency, amplitude);

    TerrainViewMode viewMode = TerrainViewMode::Lit;
    std::vector<ContourLayer> layers = terrain.extractContours(1.f);

    std::vector<std::vector<geo::GraphEdge>> adj;
    std::vector<int> path;
    bool needGenTerrain = false;
    bool weightChanged = false;
    bool scoreWeightChanged = false;
    
    rlImGuiSetup(true);

    render.runMainLoop(render::FrameCallbacks{
        [&]() { // 按键更新，重新绘图等事件
            if (debugRank != lastRank)
            {
                adj = terrain.buildAdjacencyGraph(debugRank);
                lastRank = debugRank;
            }

            // start / target 改了
            if (start != lastStart || target != lastTarget)
            {
                lastStart = start;
                lastTarget = target;
                path = terrain.shortestPathDijkstra(start, target, adj);
            }
            if (weightChanged)
            {
                adj = terrain.buildAdjacencyGraph(debugRank);
                path = terrain.shortestPathDijkstra(start, target, adj);
                weightChanged = false;
            }
            if (scoreWeightChanged && terrain.getViewMode() == TerrainViewMode::Score)
            {
                terrain.calculateInfos();
                terrain.applyFaceColor();
                scoreWeightChanged =false;
            }

            if (score_threshold != terrain.score_threshold && terrain.getViewMode() == TerrainViewMode::Score)
            {
                terrain.score_threshold = score_threshold;
                terrain.calculateInfos();
                terrain.applyFaceColor();
            }
            if (needGenTerrain)
            {
                terrain.regenerate(1 << terrainPow, 1 << terrainPow, frequency, amplitude);
                adj = terrain.buildAdjacencyGraph(debugRank);
                path = terrain.shortestPathDijkstra(start, target, adj);
                layers = terrain.extractContours(1.f);
                terrain.applyFaceColor();
                needGenTerrain = false;
            }
        },
        [&]() { // 3维空间绘图内容部分
            terrain.draw();
            terrain.drawContours(layers);
            terrain.drawPath(path, pathWidth);
            // terrain.drawGraphEdges(debugCx,debugCy,debugRank);
            //  DrawGrid(20,1.f);
            DrawLine3D({0, 0, 0}, {10000, 0, 0}, RED);
            DrawLine3D({0, 0, 0}, {0, 10000, 0}, BLUE);
            DrawLine3D({0, 0, 0}, {0, 0, -10000}, GREEN);

        },
        [&]() { // 二维屏幕空间绘图
            // render.draw_index_fonts(vertices, 16, BLUE);
            terrain.drawContourPtIndices(layers, render);
            rlImGuiBegin(); // 开始ImGui帧渲染（必须在2D阶段调用）
            // bool demoOpen = true;
            // ImGui::ShowDemoWindow(&demoOpen);

            // 2. 自定义GUI窗口（纯2D固定在屏幕上）
            bool customOpen = true;
            if (ImGui::Begin("Terrain Info", &customOpen))
            {
                ImGui::Text("Terrain size: %d x %d", 128, 128);

                // =====================================================
                // Generation Settings
                // =====================================================
                if (ImGui::CollapsingHeader("Generate Settings", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    needGenTerrain |= ImGui::SliderInt("Terrain Width(2^N)", &terrainPow, 5, 10);
                    needGenTerrain |= ImGui::SliderFloat("Frequency", &frequency, 0.f, 1.f, "%.2f");
                    needGenTerrain |= ImGui::SliderFloat("Amplitude", &amplitude, 0.f, 30.f, "%.2f");
                    needGenTerrain |= ImGui::SliderInt("Octaves",&terrain.octaves,0, 10);
                    needGenTerrain |= ImGui::SliderFloat("Lacunarity", &terrain.lacunarity, 0.f, 10.f, "%.1f");
                    needGenTerrain |= ImGui::SliderFloat("Gain", &terrain.gain, 0.f, 5.f, "%.1f");
                    if (ImGui::Button("Reset Terrain"))
                    {
                        terrain.regenerate((unsigned)time(nullptr), 1 << terrainPow, 1 << terrainPow, frequency, amplitude);
                        adj = terrain.buildAdjacencyGraph(debugRank);
                        path = terrain.shortestPathDijkstra(start, target, adj);
                        layers = terrain.extractContours(1.f);
                        terrain.applyFaceColor();
                    }
                }

                // =====================================================
                // View Settings
                // =====================================================
                if (ImGui::CollapsingHeader("View", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Text("View Mode");

                    int mode = static_cast<int>(viewMode);
                    ImGui::RadioButton("Lit", &mode, 0);
                    ImGui::RadioButton("Wire", &mode, 1);
                    ImGui::RadioButton("Aspect", &mode, 2);
                    ImGui::RadioButton("Slope", &mode, 3);
                    ImGui::RadioButton("Score", &mode, 4);

                    TerrainViewMode newMode = static_cast<TerrainViewMode>(mode);
                    if (newMode != viewMode)
                    {
                        viewMode = newMode;
                        terrain.setViewMode(viewMode);
                    }
                }

                // =====================================================
                // Draw Contours Settings
                // =====================================================
                if (ImGui::CollapsingHeader("Draw Contours"))
                {
                    terrain.buildContourSettings();
                }

                // =====================================================
                // Graph Debug
                // =====================================================
                if (ImGui::CollapsingHeader("Graph Debug"))
                {
                    ImGui::InputInt("Center X", &debugCx);
                    ImGui::InputInt("Center Y", &debugCy);
                    weightChanged |= ImGui::InputInt("Rank", &debugRank);

                    ImGui::SliderFloat("Path Width", &pathWidth, 0.05f, 1.0f, "%.2f");

                    debugCx = std::clamp(debugCx, 0, terrain.getWidth());
                    debugCy = std::clamp(debugCy, 0, terrain.getHeight());
                    debugRank = std::max(1, debugRank);

                    if (ImGui::Button("Debug Build Graph"))
                    {
                        terrain.debugBuildGraphAt(debugCx, debugCy, debugRank);
                    }
                }

                // =====================================================
                // Pathfinding
                // =====================================================
                if (ImGui::CollapsingHeader("Pathfinding"))
                {
                    ImGui::SliderInt(
                        "Start Vertex",
                        &start,
                        0, static_cast<int>(terrain.getMesh().vertices.size() - 1));

                    ImGui::SliderInt(
                        "Target Vertex",
                        &target,
                        0, static_cast<int>(terrain.getMesh().vertices.size() - 1));

                    if (ImGui::TreeNode("Cost Weights"))
                    {
                        weightChanged |= ImGui::SliderFloat("Dist Weight", &terrain.w_dist, 0.0f, 100.0f, "%.1f");
                        weightChanged |= ImGui::SliderFloat("Slope Weight", &terrain.w_slope, 0.0f, 100.0f, "%.1f");
                        weightChanged |= ImGui::SliderFloat("Up Weight", &terrain.w_up, 0.0f, 100.0f, "%.1f");
                        weightChanged |= ImGui::SliderFloat("Down Weight", &terrain.w_down, 0.0f, 100.0f, "%.1f");
                        ImGui::TreePop();
                    }
                }

                // =====================================================
                // Terrain Scoring
                // =====================================================
                if (ImGui::CollapsingHeader("Terrain Scoring"))
                {
                    scoreWeightChanged |= ImGui::SliderFloat(
                        "Seed Aspect Weight",
                        &terrain.wv_aspect, 0.f, 100.f, "%.1f");

                    scoreWeightChanged |= ImGui::SliderFloat(
                        "Seed Slope Weight",
                        &terrain.wv_slope, 0.f, 100.f, "%.1f");

                    ImGui::Separator();
                    ImGui::SliderFloat(
                        "Score Threshold",
                        &score_threshold, 0.f, 1.f, "%.2f");
                }

                // =====================================================
                // Utilities
                // =====================================================
                if (ImGui::CollapsingHeader("Utilities"))
                {
                   
                }
            }
            ImGui::End();

            rlImGuiEnd();
        }});

    return 0;
}