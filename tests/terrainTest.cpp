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
    Terrain terrain(128, 128, 1.f, 0.03f, 6.f);
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

    static float w_slope = 3.f;
    static float w_dist = 10.f;
    static float w_decent = 2.f;
    static float w_up = 0.2f;
    static float w_down = 0.2f;
    rlImGuiSetup(true);
    
    TerrainViewMode viewMode = TerrainViewMode::Lit;
    const std::vector<ContourLayer> &layers = terrain.extractContours(1.f);

    std::vector<std::vector<geo::GraphEdge>> adj;
    std::vector<int> path;
    bool weightChanged = false;
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
                path = terrain.shortestPathDijkstra(start,target,adj);
            }
            if(weightChanged){
                adj = terrain.buildAdjacencyGraph(debugRank);
                path = terrain.shortestPathDijkstra(start, target, adj);
            }
        },
        [&]() { // 3维空间绘图内容部分
            terrain.draw();
            terrain.drawContours(layers);
            terrain.drawPath(path, pathWidth);
            //terrain.drawGraphEdges(debugCx,debugCy,debugRank);
            // DrawGrid(20,1.f);
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
                ImGui::Text("terrain size: %d x %d", 128, 128);
               
                ImGui::Text("camera view : perspective");
                ImGui::Separator();
                ImGui::Text("View Mode");
                int mode = static_cast<int>(viewMode);

                if (ImGui::RadioButton("Lit", mode == 0))
                    mode = 0;
                if (ImGui::RadioButton("Wire", mode == 1))
                    mode = 1;
                if (ImGui::RadioButton("Aspect", mode == 2))
                    mode = 2;
                if (ImGui::RadioButton("Slope", mode == 3))
                    mode = 3;

                TerrainViewMode newMode = static_cast<TerrainViewMode>(mode);
                if (newMode != viewMode)
                {
                    viewMode = newMode;
                    terrain.setViewMode(viewMode);
                }

                ImGui::Separator();
                // 可添加其他UI控件（按钮、滑块等
                terrain.buildContourSettings();
                ImGui::Separator();
                if (ImGui::Button("reset terrain"))
                {
                    // 点击按钮触发地形重置逻辑
                    // terrain.reset();
                }

                ImGui::Separator();
                ImGui::Text("Graph Debug");

    
                // 输入格子坐标
                ImGui::InputInt("Center X", &debugCx);
                ImGui::InputInt("Center Y", &debugCy);
                ImGui::InputInt("Rank", &debugRank);
                ImGui::SliderFloat(
                    "Path Width",
                    &pathWidth,
                    0.05f, 1.0f,
                    "%.2f");
                // 限制范围（防止非法）
                debugCx = std::clamp(debugCx, 0, terrain.getWidth());
                debugCy = std::clamp(debugCy, 0, terrain.getHeight());
                debugRank = std::max(1, debugRank);

                ImGui::SliderInt(
                    "Start Vertex Index",
                    &start,
                    0, terrain.getMesh().vertices.size() - 1,
                    "%.0f");
                ImGui::SliderInt(
                    "Target Vertex Index",
                    &target,
                    0, terrain.getMesh().vertices.size() - 1,
                    "%.0f");
                
                // 点击按钮触发
                if (ImGui::Button("Debug Build Graph"))
                {
                    terrain.debugBuildGraphAt(debugCx, debugCy, debugRank);
                }

                weightChanged |= ImGui::SliderFloat("Dist Weight", &terrain.w_dist, 0.0f, 100.0f, "%.1f");
                weightChanged |= ImGui::SliderFloat("Slope Weight", &terrain.w_slope, 0.0f, 100.0f, "%.1f");
                weightChanged |= ImGui::SliderFloat("Up Weight", &terrain.w_up, 0.0f, 100.0f, "%.1f");
                weightChanged |= ImGui::SliderFloat("Down Weight", &terrain.w_down, 0.0f, 100.0f, "%.1f");
            }
            ImGui::End();

            rlImGuiEnd();
        }});

    return 0;
}