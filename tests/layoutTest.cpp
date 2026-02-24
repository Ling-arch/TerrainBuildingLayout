#include <iostream>
#include "geo.h"
#include "render.h"
#include "terrain.h"
#include "layout.h"
#include "tensorField.h"
#include "renderUtil.h"
#include "diffVoronoi.h"

using namespace geo;
using namespace render;
using namespace terrain;
using namespace Eigen;
using namespace layout;

int main()
{
    //-------------------------renderer and terrain init-------------------------
    bool customOpen = true;
    static bool needGenTerrain = false;
    static int terrainPow = 6;
    static float frequency = 0.01f;
    static float amplitude = 16.5f;

    rlSetClipPlanes(0.01, 100000.0);
    Renderer3D render(1920, 1080, 45.0f, CAMERA_PERSPECTIVE, "LayoutTest");
    Terrain terrain((unsigned)time(nullptr), 1 << terrainPow, 1 << terrainPow, 1.f, frequency, amplitude);
    std::vector<ContourLayer> layers = terrain.extractContours(1.f);
    TerrainViewMode viewMode = TerrainViewMode::Lit;
    terrain.setViewMode(viewMode);

    //-------------------------random polygon and tensor field-------------------------
    static int ptNum = 5;
    static float scale = 25.f;
    static float threshold = 0.5f;
    static bool genPlot = false;
    static int minGridNum = 20;
    static bool showFieldLine = true;
    static float terrainWeight = 2.f;
    bool terrainfieldWeightChanged = false;
    field::Polyline2_t<float> poly = field::createRandomPolygon(ptNum, scale, threshold, Vector2f(0.f, 0.f));
    BuildingLayout<float> layout(poly, terrain);
    field::Polyline2_t<float> rect = layout.oriRect;
    OBB2 obb(poly.points);
    // std::vector<Vector3f> projPoly;
    // std::vector<Vector3f> projRect;
    // terrain.projectPolylineToTerrain(poly.points, projPoly);
    // terrain.projectPolylineToTerrain(rect.points, projRect);
    field::TensorField2D<float> tensorField = field::TensorField2D(terrain.getAABB2(), minGridNum);
    std::unordered_map<int, field::TerrainTensor<float>> terrainTensors = terrain.sampleTensorAtGrids(tensorField.getGridPoints());
    tensorField.addConstraint({}, {}, terrainTensors);
    static float terrain_width = float(terrain.getCellSize() * terrain.getWidth());
    std::vector<Vector2f> terrainBounding2 = {
        {-terrain_width / 2.f, -terrain_width / 2.f},
        {terrain_width / 2.f, -terrain_width / 2.f},
        {terrain_width / 2.f, terrain_width / 2.f},
        {-terrain_width / 2.f, terrain_width / 2.f}};
    std::vector<Vector2f> samplePoints = M2::gen_poisson_sites_in_poly(terrainBounding2, 1.2f * tensorField.getGridSize(), 30, (unsigned)time(nullptr));
    std::vector<Polyline2_t<float>> streamlines = tensorField.genStreamlines(samplePoints);
    std::vector<RoomPoint2D<float>> roomPoints;
    std::vector<Vector2f> samplePoints2 = M2::gen_poisson_sites_in_poly(layout.rotedRect.points, layout.divGap, 30, (unsigned)time(nullptr));
    for (const auto &p : samplePoints2)
    {
        roomPoints.emplace_back(p);
    }
    RectVoronoi2D<float> rvd(roomPoints, layout.rotedBound);

    //-------------------------------diff rvd test-------------------------------
    torch::manual_seed(0);

    // ===== Example grid =====

    static bool genDiffGrid = false;
    static float beta = 20.f;
    static float tau = 10.f;
    torch::Tensor grid_xy = diffVoronoi::vec2_to_tensor(layout.rotedCenters);
    torch::Tensor terrain_h = torch::from_blob(layout.heightMap.data(), {static_cast<int64_t>(layout.heightMap.size())}).clone();
    // ===== Fixed sites =====
    torch::Tensor site_xy = diffVoronoi::vec2_to_tensor(samplePoints2);
    // ===== Model =====
    RVDModel model(grid_xy, terrain_h, site_xy, beta, tau);
    // std::cout << "h_cell before optimization:\n"
    //           << model.h_cell << std::endl;
    // torch::optim::Adam optimizer(model.parameters(), 0.05);

    // ===== Train =====
    // for (int iter = 0; iter < 200; ++iter)
    // {
    //     optimizer.zero_grad();
    //     auto loss = model.forward();
    //     loss.backward();
    //     optimizer.step();

    //     if (iter % 50 == 0)
    //     {
    //         std::cout << "Iter " << iter
    //                   << " | Loss = " << loss.item<float>()
    //                   << std::endl;
    //     }
    // }

    SoftRVDModel softModel(grid_xy, terrain_h, site_xy, beta,tau);

    rlImGuiSetup(true);
    render.runMainLoop(render::FrameCallbacks{
        [&]() { // 按键更新，重新绘图等事件，poly修改过需要重新fill
            if (genPlot)
            {
                poly = field::createRandomPolygon(ptNum, scale, threshold, Vector2f(0.f, 0.f));
                obb = OBB2(poly.points);
                // terrain.projectPolylineToTerrain(poly.points, projPoly);
                // terrain.projectPolylineToTerrain(rect.points, projRect);
                layout = BuildingLayout<float>(poly, terrain);
                rect = layout.oriRect;
                samplePoints2 = M2::gen_poisson_sites_in_poly(layout.rotedRect.points, layout.divGap, 30, (unsigned)time(nullptr));
                roomPoints.clear();
                for (const auto &p : samplePoints2)
                {
                    roomPoints.emplace_back(p);
                }
                grid_xy = diffVoronoi::vec2_to_tensor(layout.rotedCenters);
                site_xy = diffVoronoi::vec2_to_tensor(samplePoints2);
                terrain_h = torch::from_blob(layout.heightMap.data(), {static_cast<int64_t>(layout.heightMap.size())}).clone();
                rvd = RectVoronoi2D<float>(roomPoints, layout.rotedBound);
                softModel = SoftRVDModel(grid_xy, terrain_h, site_xy, beta, tau);
                genPlot = false;
            }

            if (needGenTerrain)
            {
                terrain.regenerate(1 << terrainPow, 1 << terrainPow, frequency, amplitude);
                terrain.applyFaceColor();
                layers = terrain.extractContours(1.f);
                tensorField = field::TensorField2D(terrain.getAABB2(), minGridNum);
                terrainTensors = terrain.sampleTensorAtGrids(tensorField.getGridPoints());
                tensorField.addConstraint({}, {}, terrainTensors);
                samplePoints = M2::gen_poisson_sites_in_poly(terrainBounding2, 1.2f * tensorField.getGridSize(), 30, (unsigned)time(nullptr));
                streamlines = tensorField.genStreamlines(samplePoints);
                layout = BuildingLayout<float>(poly, terrain);
                rect = layout.oriRect;
                samplePoints2 = M2::gen_poisson_sites_in_poly(layout.rotedRect.points, layout.divGap, 30, (unsigned)time(nullptr));
                roomPoints.clear();
                for (const auto &p : samplePoints2)
                {
                    roomPoints.emplace_back(p);
                }
                rvd = RectVoronoi2D<float>(roomPoints, layout.rotedBound);
                grid_xy = diffVoronoi::vec2_to_tensor(layout.rotedCenters);
                site_xy = diffVoronoi::vec2_to_tensor(samplePoints2);
                terrain_h = torch::from_blob(layout.heightMap.data(), {static_cast<int64_t>(layout.heightMap.size())}).clone();
                softModel = SoftRVDModel(grid_xy, terrain_h, site_xy, beta, tau);
                needGenTerrain = false;
            }

            if (terrainfieldWeightChanged)
            {
                tensorField.setTensorWeight(terrainWeight);
                streamlines = tensorField.genStreamlines(samplePoints);
                terrainfieldWeightChanged = false;
            }

            if (genDiffGrid)
            {
                model = RVDModel(grid_xy, terrain_h, site_xy, beta);
                softModel = SoftRVDModel(grid_xy, terrain_h, site_xy, beta, tau);
                genDiffGrid = false;
            }
        },
        [&]() { // 3维空间绘图内容部分
            //  terrain.draw();
            //  layout.drawTerrain(RL_GRAY, 0.8f, true, 0.5f);
            terrain.drawContours(layers);
            // DrawLine3D({0, 0, 0}, {10000, 0, 0}, RL_RED);
            // DrawLine3D({0, 0, 0}, {0, 10000, 0}, RL_BLUE);
            // DrawLine3D({0, 0, 0}, {0, 0, -10000}, RL_GREEN);
            // render::stroke_bold_polygon2(poly.points, RL_BLACK, 0.f, 0.07f, 1.f);
            // render::stroke_bold_polygon2(rect.points, RL_RED, 0.f, 0.07f, 1.f);
            // render::stroke_bold_polygon2(obb.poly.points, RL_RED, 0.f, 0.07f, 1.f);
            // render::stroke_bold_polygon3(projPoly,RL_BLACK,0.03F);
            // render::stroke_bold_polygon3(projRect,RL_RED,0.03F);
            if (showFieldLine)
            {
                for (const auto &line : streamlines)
                {
                    render::draw_bold_polyline2(line.points, render.lineData.color, 0.f, render.lineData.Thickness, render.lineData.color.a);
                }
                for (const field::Tensor<float> &t : tensorField.getAllTensors())
                {
                    for (int i = 0; i < 4; ++i)
                        render::draw_vector(t.pos, t.dirs[i], render.vecData.color, render.vecData.scale, render.vecData.startThickness, render.vecData.endThickness, render.vecData.vecZ, render.vecData.color.a);
                }
            }

            for (int i = 0; i < rvd.getCells().size(); ++i)
            {
                // const geo::Polyline2_t<float> &cellPoly = geo::rotatePoly(rvd.getCellPolys()[i], layout.Rinv);
                Color c = renderUtil::ColorFromHue((float)i / rvd.getCellPolys().size());
                // render::fill_polygon2(cellPoly.points, c, 0.f, 0.3f);
                render::fill_polygon2(rvd.getCellPolys()[i].points, c, 0.f, 0.3f, {terrain_width / 2.f, 0});
                render::stroke_bold_polygon2(rvd.getCellPolys()[i].points, RL_BLACK, 0.f, 0.07f, 1.f, {terrain_width / 2.f, 0});
            }
            // render::draw_points(layout.rotedCenters, render.ptData.color, 1.f, render.ptData.size / 2.f);
            // render::draw_points(samplePoints2, render.ptData.color, 1.f, render.ptData.size * 1.5f, 0, {terrain_width / 2.f, 0.f});
            model.drawGrids();
            softModel.drawGrids(0.f, 1.f, {terrain_width, 0.f});
        },
        [&]() { // 二维屏幕空间绘图
            // render.draw_index_fonts(layout.rotedCenters, render.ptData.size, render.ptData.color, 0,  {terrain_width, 0.f});
            // render.draw_index_fonts(samplePoints2, render.ptData.size*2, RL_RED, 0,{terrain_width / 2.f, 0.f});
            // render.draw_index_fonts(layout.rotedGrids, render.ptData.size, render.ptData.color);
            rlImGuiBegin();

            render.setCameraUI(customOpen);
            render.setDrawGeoDataUI(customOpen);
            if (ImGui::Begin("Terrain Info", &customOpen))
            {
                ImGui::Text("Terrain size: %d x %d", 128, 128);

                // =====================================================
                // Generation Settings
                // =====================================================
                if (ImGui::CollapsingHeader("Generate Settings", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Indent();
                    needGenTerrain |= ImGui::SliderInt("Terrain Width(2^N)", &terrainPow, 5, 11);
                    needGenTerrain |= ImGui::SliderFloat("Frequency", &frequency, 0.f, 1.f, "%.2f");
                    needGenTerrain |= ImGui::SliderFloat("Amplitude", &amplitude, 0.f, 30.f, "%.2f");
                    needGenTerrain |= ImGui::SliderInt("Octaves", &terrain.octaves, 0, 10);
                    needGenTerrain |= ImGui::SliderFloat("Lacunarity", &terrain.lacunarity, 0.f, 10.f, "%.1f");
                    needGenTerrain |= ImGui::SliderFloat("Gain", &terrain.gain, 0.f, 5.f, "%.1f");
                    needGenTerrain |= ImGui::Button("Reset Terrain");
                    ImGui::Unindent();
                }
                // =====================================================
                // View Settings
                // =====================================================
                if (ImGui::CollapsingHeader("View", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Indent();
                    ImGui::Text("View Mode");

                    int mode = static_cast<int>(viewMode);
                    ImGui::Checkbox("AdditionalWire", &terrain.additionalShowWire);
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
                    ImGui::Unindent();
                }

                // =====================================================
                // Draw Contours Settings
                // =====================================================
                if (ImGui::CollapsingHeader("Draw Contours"))
                {
                    ImGui::Indent();
                    terrain.buildContourSettings();
                    ImGui::Unindent();
                }
            }
            ImGui::End();

            if (ImGui::Begin("TensorField", &customOpen))
            {
                if (ImGui::CollapsingHeader("PlotGenSetting", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Indent();
                    genPlot |= ImGui::SliderInt("PtNum", &ptNum, 4, 12);
                    genPlot |= ImGui::SliderFloat("Scale", &scale, 0.f, 2000.f, "%.1f");
                    genPlot |= ImGui::SliderFloat("Threshold", &threshold, 0.f, 1.f, "%.2f");
                    terrainfieldWeightChanged |= ImGui::SliderFloat("Terrain Field Weight", &terrainWeight, 1.f, 10.f, "%.1f");
                    ImGui::Checkbox("FieldShow", &showFieldLine);
                    ImGui::Unindent();
                }
            }
            ImGui::End();

            if (ImGui::Begin("DiffRVD", &customOpen))
            {
                if (ImGui::CollapsingHeader("DrawGrids", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Indent();
                    genDiffGrid |= ImGui::SliderFloat("Beta", &beta, 5.f, 100.f, "%.1f");
                    genDiffGrid |= ImGui::SliderFloat("Tau", &tau, 0.1f, 100.f, "%.1f");
                    ImGui::Unindent();
                }
            }
            ImGui::End();
            rlImGuiEnd();
        }});
    return 0;
}