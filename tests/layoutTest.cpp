#include <iostream>
#include "geo.h"
#include "render.h"
#include "terrain.h"
#include "layout.h"
#include "tensorField.h"
#include "renderUtil.h"
#include "diffVoronoi.h"
#include "SCARoadGenerator.h"
#include "grid.h"
using namespace geo;
using namespace render;
using namespace terrain;
using namespace Eigen;
using namespace layout;
using namespace SCARoad;
using namespace grid;

int main()
{
    //-------------------------renderer and terrain init-------------------------
    bool customOpen = true;
    static bool needGenTerrain = false;
    static int terrainPow = 8;
    static float frequency = 0.02f;
    static float amplitude = 21.5f;

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
    static bool showFieldLine = false;
    static float terrainWeight = 2.f;
    bool terrainfieldWeightChanged = false;
    static bool showTerrain = false;
    bool viewPtChanged = false;
    bool showViewPt = false;

    static int debugNodeID_1 = 0;
    static float debugRadius = 20.0f;

    static int debugNodeID_2 = 0;
    static int debugNodeID_3 = 0;

    static bool showSoftRVDGrids = true;
    static bool showFinalRVDVolume = false;
    static bool isFirstSoftIter = true;
    field::Polyline2_t<float> poly = field::createRandomPolygon(ptNum, scale, threshold, Vector2f(0.f, 0.f));
    BuildingLayout<float> layout(poly, terrain);
    field::Polyline2_t<float> rect = layout.oriRect;
    OBB2 obb(poly.points);
    std::vector<Vector3f> projPoly;
    std::vector<Vector3f> projRect;
    terrain.projectPolylineToTerrain(poly.points, projPoly);
    terrain.projectPolylineToTerrain(rect.points, projRect);
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

    Polyline2_t<float> boundOffset = geo::offsetPolygon(layout.rotedRect, -2.f)[0];
    std::vector<Vector2f> yardSeeds = geo::samplePointsOnPolygonWithSpacing(boundOffset, 2, (unsigned)time(nullptr));
    M2::PoissonResult totalSeedResult = M2::gen_poisson_sites_in_poly_with_seeds(layout.rotedRect.points, yardSeeds, layout.divGap, 5, 30, (unsigned)time(nullptr));
    //-------------------------------diff rvd test-------------------------------

    torch::manual_seed(0);

    // ===== Example grid =====

    static bool genDiffGrid = false;
    static float beta = 20.f;
    static float tau = 10.f;
    torch::Tensor grid_xy = diffVoronoi::vec2_to_tensor(layout.rotedCenters);
    torch::Tensor terrain_h = torch::from_blob(layout.heightMap.data(), {static_cast<int64_t>(layout.heightMap.size())},
                                               torch::kFloat32) // 显式指定类型（和 heightMap 一致）
                                  .clone()                      // 深拷贝，避免原数据释放后失效
                                  .to(torch::kCPU);             // 转到 CPU（可视化必须在 CPU 读取）).clone();
    // ===== Fixed sites =====
    torch::Tensor site_xy = diffVoronoi::vec2_to_tensor(totalSeedResult.samples);

    SoftRVDModel softModel(grid_xy, terrain_h, site_xy, 1, {0, 1}, {1,1,0,1,1},beta, tau);
    SoftRVDShowData showData;
    std::cout << "softmodel built" << std::endl;
    // softModel.optimizeLloyd();
    bool isOptimizing = false;
    static int maxIter = 200;
    int curIter = 0;

    Polyline2_t<float> testGridPoly({{0.f, 0.f},
                                     {18.f, 0.f},
                                     {18.f, 8.f},
                                     {12.f, 8.f},
                                     {12.f, 15.f},
                                     {4.f, 15.f},
                                     {4.f, 8.f},
                                     {0.f, 8.f}},
                                    true);
    CellGenerator cellGen(testGridPoly, 1.f);
    std::vector<int> indices;
    indices.reserve(cellGen.cells.size());

    for (int i = 0; i < cellGen.cells.size(); ++i)
    {
        indices.push_back(i);
    }
    std::cout << "Total cells: " << cellGen.cells.size() << std::endl;
    CellGroup cellGroup(indices, &cellGen.cells, 0);
  
    srand((unsigned)time(nullptr));

    rlImGuiSetup(true);
    render.runMainLoop(render::FrameCallbacks{
        [&]() { // 按键更新，重新绘图等事件，poly修改过需要重新fill
            if (genPlot)
            {
                poly = field::createRandomPolygon(ptNum, scale, threshold, Vector2f(0.f, 0.f));
                obb = OBB2(poly.points);
                terrain.projectPolylineToTerrain(poly.points, projPoly);
                terrain.projectPolylineToTerrain(rect.points, projRect);
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
                softModel = SoftRVDModel(grid_xy, terrain_h, site_xy, 1, {0, 1}, {1, 1, 0, 1, 1}, beta, tau);
                genPlot = false;
            }

            if (needGenTerrain)
            {
                terrain.regenerate(1 << terrainPow, 1 << terrainPow, frequency, amplitude);
                terrain.applyFaceColor();
                terrain_width = float(terrain.getCellSize() * terrain.getWidth());
                layers = terrain.extractContours(1.f);
                tensorField = field::TensorField2D(terrain.getAABB2(), minGridNum);
                terrainTensors = terrain.sampleTensorAtGrids(tensorField.getGridPoints());
                tensorField.addConstraint({}, {}, terrainTensors);
                samplePoints = M2::gen_poisson_sites_in_poly(terrainBounding2, 1.2f * tensorField.getGridSize(), 30, (unsigned)time(nullptr));
                streamlines = tensorField.genStreamlines(samplePoints);
                layout = BuildingLayout<float>(poly, terrain);
                rect = layout.oriRect;
                totalSeedResult = M2::gen_poisson_sites_in_poly_with_seeds(layout.rotedRect.points, yardSeeds, layout.divGap, 5, 30, (unsigned)time(nullptr));
                roomPoints.clear();
                for (const auto &p : totalSeedResult.samples)
                {
                    roomPoints.emplace_back(p);
                }
                rvd = RectVoronoi2D<float>(roomPoints, layout.rotedBound);
                grid_xy = diffVoronoi::vec2_to_tensor(layout.rotedCenters);
                site_xy = diffVoronoi::vec2_to_tensor(totalSeedResult.samples);
                terrain_h = torch::from_blob(layout.heightMap.data(), {static_cast<int64_t>(layout.heightMap.size())}).clone();
                softModel = SoftRVDModel(grid_xy, terrain_h, site_xy, 1, {0, 1}, {1, 1, 0, 1, 1}, beta, tau);
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
                // model = RVDModel(grid_xy, terrain_h, site_xy, beta);
                softModel = SoftRVDModel(grid_xy, terrain_h, site_xy, 1, {0, 1}, {1, 1, 0, 1, 1}, beta, tau);
                genDiffGrid = false;
            }

            if (IsKeyPressed(KEY_A))
            {
                if (isFirstSoftIter){
                    isOptimizing = true;
                }else{

                    isOptimizing = true;
                    curIter = 0;
                    totalSeedResult = M2::gen_poisson_sites_in_poly_with_seeds(layout.rotedRect.points, yardSeeds, layout.divGap, 5, 30, (unsigned)time(nullptr));
                    grid_xy = diffVoronoi::vec2_to_tensor(layout.rotedCenters);
                    site_xy = diffVoronoi::vec2_to_tensor(totalSeedResult.samples);
                    terrain_h = torch::from_blob(layout.heightMap.data(), {static_cast<int64_t>(layout.heightMap.size())}).clone();
                    softModel = SoftRVDModel(grid_xy, terrain_h, site_xy, 1, {0, 1}, {1, 1, 0, 1, 1}, beta, tau);
                }
              
               
            }
            if (isOptimizing)
            {
                softModel.stepOptimize(showData, curIter, maxIter, isOptimizing);
            }
            if (curIter >= maxIter)
            {
                isOptimizing = false;
                curIter = 0;
            }

            if (viewPtChanged)
            {
                terrain.applyFaceColor();
            }

        },
        [&]() { // 3维空间绘图内容部分
            if (showTerrain)
                terrain.draw();
            layout.drawTerrain(RL_GRAY, 0.8f, true, 0.5f);
            // DrawGrid(50,10);
            terrain.drawContours(layers);
            // DrawSphere({0, 0, 0}, 2.f, RL_RED);
            if (showViewPt)
            {
                DrawSphere({terrain.testViewPt.x(), terrain.observeHeight, -terrain.testViewPt.y()}, 1.f, RL_GRAY);
            }

            showData.draw(4.f,1.f/* ,{terrain_width / 4.f, 0.f} */);
            DrawLine3D({0, 0, 0}, {10000, 0, 0}, RL_RED);
            DrawLine3D({0, 0, 0}, {0, 10000, 0}, RL_BLUE);
            DrawLine3D({0, 0, 0}, {0, 0, -10000}, RL_GREEN);
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

            // for (int i = 0; i < rvd.getCells().size(); ++i)
            // {
            //     // const geo::Polyline2_t<float> &cellPoly = geo::rotatePoly(rvd.getCellPolys()[i], layout.Rinv);
            //     Color c = renderUtil::ColorFromHue((float)i / rvd.getCellPolys().size());
            //     // render::fill_polygon2(cellPoly.points, c, 0.f, 0.3f);
            //     render::fill_polygon2(rvd.getCellPolys()[i].points, c, 0.f, 0.3f, {terrain_width / 2.f, 0});
            //     render::stroke_bold_polygon2(rvd.getCellPolys()[i].points, RL_BLACK, 0.f, 0.07f, 1.f, {terrain_width / 2.f, 0});
            // }
            //  render::draw_points(samplePoints2, render.ptData.color, 1.f, render.ptData.size / 2.f,0.f,{terrain_width / 2.f, 0});
            // render::draw_points(layout.rotedCenters, render.ptData.color, 1.f, render.ptData.size / 2.f);
            // render::draw_points(layout.sampleCenters, RL_RED, 1.f, render.ptData.size);
            // render::draw_points(layout.grids, render.ptData.color, 1.f, render.ptData.size / 1.5f);
            // render::stroke_bold_polygon2(boundOffset.points, RL_BLACK, 0.f, 0.07f, 1.f, {terrain_width / 2.f, 0});
            // render::draw_points(totalSeedResult.samples, render.ptData.color, 1.f, render.ptData.size * 1.5f, 0, {terrain_width / 2.f, 0.f});

            // model.drawGrids();
            softModel.drawGrids(0.f, 1.f, {terrain_width / 4.f, 0.f});
            // softModel.drawTerrain(layout.heightMap);
        },
        [&]() { // 二维屏幕空间绘图
            // render.draw_index_fonts(layout.rotedCenters, render.ptData.size, render.ptData.color, 0, {terrain_width / 2.f, 0.f});
            render.draw_index_fonts(totalSeedResult.samples, render.fontData.size * 2, RL_RED, 0, {terrain_width / 2.f, 0.f});
            // render.draw_index_fonts(layout.rotedGrids, render.ptData.size, render.ptData.color);
            // net.drawNodesWithIndices(render);

            rlImGuiBegin();

            render.setCameraUI(customOpen);
            render.setDrawGeoDataUI(customOpen);
            if (ImGui::Begin("Terrain Info", &customOpen))
            {
                ImGui::Checkbox("Show Terrain", &showTerrain);
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
                    ImGui::RadioButton("ViewShed", &mode, 4);
                    ImGui::RadioButton("PointViewShed", &mode, 5);
                    ImGui::Indent();
                    ImGui::Checkbox("ShowViewPt", &terrain.showTestViewPt);
                    viewPtChanged |= ImGui::SliderFloat("ViewPtX", &terrain.testViewPt.x(), -terrain_width / 2.f, terrain_width / 2.f, "%.1f");
                    viewPtChanged |= ImGui::SliderFloat("ViewPtY", &terrain.testViewPt.y(), -terrain_width / 2.f, terrain_width / 2.f, "%.1f");
                    viewPtChanged |= ImGui::SliderFloat("ObserveH", &terrain.observeHeight, 0.f, 30.f, "%.1f");
                    ImGui::Unindent();
                    ImGui::RadioButton("Flow", &mode, 6);
                    ImGui::RadioButton("Score", &mode, 7);

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

            if (ImGui::Begin("SCADebug", &customOpen))
            {
                ImGui::Indent();

                // Debug 1: Relative Neighbors
                ImGui::Separator();
                ImGui::Text("Relative Neighbor Query");

                ImGui::InputInt("Node ID##Neighbor", &debugNodeID_1);
                ImGui::InputFloat("Radius", &debugRadius);

                // Debug 2: Path Info
                ImGui::Separator();
                ImGui::Text("Node Path Info");
                ImGui::InputInt("Node ID##Path", &debugNodeID_2);
                ImGui::Separator();
                ImGui::Text("Connect Info");
                ImGui::InputInt("Node ID##ConnectNode", &debugNodeID_3);

                ImGui::Unindent();
            }
            ImGui::End();
            rlImGuiEnd();
        }});
    return 0;
}