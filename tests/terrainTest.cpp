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
    std::vector<Vector3> vertices = terrain.getMeshVertices();
    rlImGuiSetup(true);
    bool disableCamera = false;
    TerrainViewMode viewMode = TerrainViewMode::Lit;
    std::vector<ContourLayer> layers = terrain.extractContours(1.f);
    for (ContourLayer &layer : layers)
    {
        std::vector<geo::Segment> segments = layer.segments;
        std::vector<geo::Polyline> polylines = buildPolylines(segments);
        for (int i = 0; i < polylines.size(); ++i)
        {
            std::cout << "Polyline " << i << " ("
                      << polylines[i].points.size() << " pts)\n";

            bool closed = polylines[i].closed;

            std::cout << "  closed: " << (closed ? "YES" : "NO") << "\n";

            for (auto &p : polylines[i].points)
            {
                std::cout << "    (" << p.x()
                          << "," << p.y()
                          << "," << p.z() << ")\n";
            }
        }
    }

    render.runMainLoop(render::FrameCallbacks{
        [&]() { // 按键更新，重新绘图等事件

        },
        [&]() { // 3维空间绘图内容部分
            terrain.draw();
            terrain.drawContours();
            // DrawGrid(20,1.f);
            DrawLine3D({0, 0, 0}, {10000, 0, 0}, RED);
            DrawLine3D({0, 0, 0}, {0, 10000, 0}, BLUE);
            DrawLine3D({0, 0, 0}, {0, 0, -10000}, GREEN);
        },
        [&]() { // 二维屏幕空间绘图
            // render.draw_index_fonts(vertices, 16, BLUE);

            rlImGuiBegin(); // 开始ImGui帧渲染（必须在2D阶段调用）
            // bool demoOpen = true;
            // ImGui::ShowDemoWindow(&demoOpen);

            // 2. 自定义GUI窗口（纯2D固定在屏幕上）
            bool customOpen = true;
            if (ImGui::Begin("Terrain Info", &customOpen))
            {
                ImGui::Text("terrain size: %d x %d", 128, 128);
                ImGui::Text("vertices num : %zu", vertices.size());
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
                if (ImGui::Button("Draw Contours"))
                {
                    if (!terrain.getContousShow())
                        terrain.setContoursShow(true);
                    else
                        terrain.setContoursShow(false);
                }
                if (ImGui::Button("reset terrain"))
                {
                    // 点击按钮触发地形重置逻辑
                    // terrain.reset();
                }
            }
            ImGui::End();

            rlImGuiEnd();
        }});

    return 0;
}