#include <iostream>
#include "render.h"

#include "terrain.h"
#include "renderUtil.h"
#include <rlImGui.h>
#include <imgui.h>
#include "tensorField.h"
#include "optimizer.h"

using Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector2d;
using field::TensorField2D, field::PointAtrractor, field::TerrainTensor;
using render::Renderer3D;
using terrain::Terrain, terrain::TerrainCell, terrain::TerrainViewMode, terrain::ContourLayer, terrain::Road;
using namespace render;
using namespace optimizer;
using namespace geo;

int main()
{
    rlSetClipPlanes(0.01, 100000.0);
    std::cout << "Hello, TerrainTest Start!" << std::endl;
    Renderer3D render(1920, 1080, 45.0f, CAMERA_PERSPECTIVE, "TerrainTest");
    render::RENDER_ZOOM_SPEED = 8.f;
    render::RENDER_MOVE_SPEED = 1.f;
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
    static int terrainPow = 7;
    static float frequency = 0.01f;
    static float amplitude = 23.5f;
    static int mainRoadNode = 16;
    int lastMainRoadNode = -1;
    static float extrudeHeight = 2.f;
    float lastExtrudeHeight = -1.f;
    static float colorAlpha = 0.5f;
    static float wireframeAlpha = 0.8f;
    bool wireframe = true;
    bool outline = true;

    PointDrawData ptData;
    VectorDrawData vecData;
    FontDrawData fontData;
    LineDrawData lineData;
    static float vecZ = 0.f;
    static int minGridNum = 30;
    static float terrainWeight = 2.f;
    bool genTensorfield = false;
    bool terrainfieldWeightChanged = false;
    bool showFieldLine = true;
    bool showText = false;

    Eigen::Vector2i roadGapRange = {18, 55};
    Eigen::Vector2i depthRange = {18, 55};
    Terrain terrain((unsigned)time(nullptr), 1 << terrainPow, 1 << terrainPow, 2.f, frequency, amplitude);

    TerrainViewMode viewMode = TerrainViewMode::Lit;
    terrain.setViewMode(viewMode);
    std::vector<ContourLayer> layers = terrain.extractContours(1.f);

    std::vector<std::vector<geo::GraphEdge>> adj;
    // std::vector<int> path;
    bool needGenTerrain = false;
    bool weightChanged = false;
    bool scoreWeightChanged = false;
    std::vector<Road> mainPaths;
    std::vector<Polyline2_t<float>> pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
    std::vector<Eigen::Vector3f> seedPoints;
    std::vector<Eigen::Vector3f> roadControlPts;
    static float radius = 10.f;
    bool radiusChanged = false;
    std::vector<PointAtrractor<float>> attractors;
    attractors.emplace_back(Vector2f(0.f, 0.f), radius);
    TensorField2D<float> tensorField = TensorField2D(terrain.getAABB2(), minGridNum);
    std::unordered_map<int, field::TerrainTensor<float>> terrainTensors = terrain.sampleTensorAtGrids(tensorField.getGridPoints());
    tensorField.addConstraint(pathPolylines, attractors, terrainTensors);
    std::vector<Vector2f> terrainBounding2 = {
        {-terrain_width / 2.f, -terrain_width / 2.f},
        {terrain_width / 2.f, -terrain_width / 2.f},
        {terrain_width / 2.f, terrain_width / 2.f},
        {-terrain_width / 2.f, terrain_width / 2.f}};
    std::vector<Vector2f> samplePoints = M2::gen_poisson_sites_in_poly(terrainBounding2, 1.2f * tensorField.getGridSize(), 30, (unsigned)time(nullptr));
    std::vector<Polyline2_t<float>> streamlines = tensorField.genStreamlines(samplePoints);
    std::vector<std::vector<Vector3f>> projStreamlines;
    for (const auto &l : streamlines)
    {
        std::vector<Vector3f> projLine;
        if (terrain.projectPolylineToTerrain(l.points, projLine))
            projStreamlines.push_back(projLine);
    }

    std::vector<std::vector<Vector3f>> projParcels;
    std::vector<Polyline2_t<float>> parcels = tensorField.trace2sidesParcels((unsigned)time(nullptr), 1.2f, 1.5f, roadGapRange);
    std::vector<Polyline2_t<float>> parcelObbs;
    for (const auto &polyline : parcels)
    {
        std::vector<Vector3f> projLine;
        if (terrain.projectPolylineToTerrain(polyline.points, projLine))
        {
            projParcels.push_back(projLine);
        }
        geo::OBB2<float> obb(polyline.points);
        parcelObbs.push_back(obb.poly);
    }

    Polyline2_t<float> testPolyA({{0.f, 0.f}, {10.f, 10.f}, {20.f, 0.f}, {20.f, 20.f}, {0.f, 20.f}, {0.f, 0.f}}, true);
    Polyline2_t<float> testPolyB({{10.f, 0.f}, {31.f, 11.f}, {12.f, 14.f}, {10.f, 0.f}}, true);
    std::vector<Vector2f> cutPoints = {{0.f, 10.f}, {20.f, 10.f}, {10.f, 30.f}, {-5.f, 10.f}};
    // cutPoints.insert(cutPoints.end(), std::make_move_iterator(cutPoints.end()), std::make_move_iterator(cutPoints.begin()));
    Polyline2_t<float> cutline(cutPoints);

    Polyline2_t<float> unionPoly = geo::unionPolygon(testPolyA, testPolyB);
    std::vector<Polyline2_t<float>> subedPolys = geo::subPolygon(testPolyA, testPolyB);
    std::vector<Polyline2_t<float>> interPolys = geo::intersectPolygon(testPolyA, testPolyB);
    std::vector<Polyline2_t<float>> cuttedPolys = geo::splitPolygonByPolylines(testPolyA, {cutline});
    std::cout << "cuttedpolys is " << cuttedPolys.size() << std::endl;
    std::vector<Polyline2_t<float>> offsetPolys;
    static float offsetDist = -1.f;
    bool offsetPoly = false;
    for (const auto &poly : cuttedPolys)
    {
        std::vector<Polyline2_t<float>> offsets = geo::offsetPolygon(poly, offsetDist);
        offsetPolys.insert(offsetPolys.end(), offsets.begin(), offsets.end());
    }

    const std::vector<float> area_ratio = {0.4f, 0.3f, 0.2f, 0.1f, 0.1f, 0.1f};
    const std::vector<std::pair<size_t, size_t>> room_connections = {{0, 1}, {0, 2}, {1, 3}, {0, 4}, {4, 5}};
    PlanProblem plan_prob;
    std::vector<Vector2f> testParcelBounds;
    if (parcels.size() > 0)
    {
        plan_prob = define_field_problem(0, tensorField, parcels[0].points, area_ratio, room_connections, {}, {});
        testParcelBounds = polyloop::denormalize_to_pts(plan_prob.vtxl2xy_norm, plan_prob.tf);
    }

    // VoronoiResult2 result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
    // std::cout << "cell size is " << result.voronoi_cells.size() << std::endl;
    std::vector<Color> room2colors = renderUtil::room2colors(area_ratio.size());
    OptimizeDrawData draw_data;
    size_t cur_iter = 0;
    bool is_optimizing = false;
    // geo::PolygonMesh extrudeMesh = geo::PolygonMesh({{0, 0, 0}, {30, 0, 0}, {30, 15, 0}, {45, 15, 0}, {45, 30, 0}, {0, 30, 0}}, extrudeHeight);
    // geo::PolygonMesh extrudeMesh_2 = geo::PolygonMesh({{0, 0, extrudeHeight}, {30, 0, extrudeHeight}, {30, 30, extrudeHeight}, {0, 30, extrudeHeight}}, extrudeHeight);
    // geo::PolygonMesh extrudeMesh_3 = geo::PolygonMesh({{30, 0, 0}, {45, 0, 0}, {45, 15, 0}, {30, 15, 0}}, extrudeHeight);
    // polyloop::Polyloop3 polyloop({{0, 0,0}, {30, 0,0}, {30, 30,0}, {0, 30,0}});
    Color color = RL_YELLOW;
    Ray ray = {0};
    rlImGuiSetup(true);

    auto terrainRebuild = [&]() {

    };

    auto tensorFieldRebuild = [&]
    {
        tensorField.addConstraint(pathPolylines, attractors, terrainTensors);
        samplePoints = M2::gen_poisson_sites_in_poly(terrainBounding2, 1.2f * tensorField.getGridSize(), 30, (unsigned)time(nullptr));
        streamlines = tensorField.genStreamlines(samplePoints);
        projStreamlines.clear();
        for (const auto &l : streamlines)
        {
            std::vector<Vector3f> projLine;
            if (terrain.projectPolylineToTerrain(l.points, projLine))
                projStreamlines.push_back(projLine);
        }
    };

    auto parcelsRebuild = [&]
    {
        parcels = tensorField.trace2sidesParcels((unsigned)time(nullptr), 1.2f, 1.5f, roadGapRange);
        projParcels.clear();
        parcelObbs.clear();
        for (const auto &polyline : parcels)
        {
            std::vector<Vector3f> projLine;
            if (terrain.projectPolylineToTerrain(polyline.points, projLine))
                projParcels.push_back(projLine);
            geo::OBB2<float> obb(polyline.points);
            parcelObbs.push_back(obb.poly);
        }
    };

    render.runMainLoop(render::FrameCallbacks{
        [&]() { // 按键更新，重新绘图等事件
            if (IsKeyPressed(KEY_R) && parcels.size() > 0)
            {
                plan_prob = define_field_problem(0, tensorField, parcels[0].points, area_ratio, room_connections, {}, {});
                testParcelBounds = polyloop::denormalize_to_pts(plan_prob.vtxl2xy_norm, plan_prob.tf);
                is_optimizing = true;
                cur_iter = 0;
            }

            if (is_optimizing)
            {
                optimize_field_problem_and_draw_bystep(plan_prob, cur_iter, 350, draw_data);
            }
            if (debugRank != lastRank)
            {
                adj = terrain.buildAdjacencyGraph(debugRank);
                lastRank = debugRank;
            }

            if (weightChanged)
            {
                adj = terrain.buildAdjacencyGraph(debugRank);
                mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);
                pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
                //result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                weightChanged = false;
            }
            if (scoreWeightChanged && terrain.getViewMode() == TerrainViewMode::Score)
            {
                terrain.calculateInfos();
                terrain.applyFaceColor();
                mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);
                pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
                //result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                // std::cout << "polylines num is " << pathPolylines.size() << std::endl;
                tensorFieldRebuild();
                parcelsRebuild();
                scoreWeightChanged = false;
            }

            if (score_threshold != terrain.score_threshold && terrain.getViewMode() == TerrainViewMode::Score)
            {
                terrain.score_threshold = score_threshold;
                terrain.calculateInfos();
                terrain.applyFaceColor();
                mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);
                pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
                //result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                tensorFieldRebuild();
                parcelsRebuild();
            }

            if (mainRoadNode != lastMainRoadNode)
            {
                lastMainRoadNode = mainRoadNode;
                mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);
                pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
                //result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                tensorFieldRebuild();
                parcelsRebuild();
            }
            if (needGenTerrain)
            {
                terrain.regenerate(1 << terrainPow, 1 << terrainPow, frequency, amplitude);
                adj = terrain.buildAdjacencyGraph(debugRank);
                mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);
                pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
                //result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                layers = terrain.extractContours(1.f);
                terrain.applyFaceColor();
                tensorField = TensorField2D(terrain.getAABB2(), minGridNum);
                terrainTensors = terrain.sampleTensorAtGrids(tensorField.getGridPoints());
                tensorFieldRebuild();
                parcelsRebuild();
                needGenTerrain = false;
            }

            if (extrudeHeight != lastExtrudeHeight)
            {
                lastExtrudeHeight = extrudeHeight;
                // extrudeMesh.regenerate(extrudeHeight);
                // extrudeMesh_2 = geo::PolygonMesh({{0, 0, extrudeHeight}, {30, 0, extrudeHeight}, {30, 30, extrudeHeight}, {0, 30, extrudeHeight}}, extrudeHeight);
                // extrudeMesh_3.regenerate(extrudeHeight);
            }

            if (genTensorfield)
            {
                tensorField = TensorField2D(terrain.getAABB2(), minGridNum);
                terrainTensors = terrain.sampleTensorAtGrids(tensorField.getGridPoints());
                tensorFieldRebuild();
                parcelsRebuild();
                genTensorfield = false;
            }

            if (terrainfieldWeightChanged)
            {
                tensorField.setTensorWeight(terrainWeight);
                streamlines = tensorField.genStreamlines(samplePoints);
                projStreamlines.clear();
                for (const auto &l : streamlines)
                {
                    std::vector<Vector3f> projLine;
                    if (terrain.projectPolylineToTerrain(l.points, projLine))
                        projStreamlines.push_back(projLine);
                }
                parcelsRebuild();
                terrainfieldWeightChanged = false;
            }

            if (radiusChanged)
            {
                for (PointAtrractor<float> &pa : tensorField.getAttractorsRef())
                {
                    pa.radius = radius;
                }
                tensorField.resolveTensor();
                streamlines = tensorField.genStreamlines(samplePoints);
                projStreamlines.clear();
                for (const auto &l : streamlines)
                {
                    std::vector<Vector3f> projLine;
                    if (terrain.projectPolylineToTerrain(l.points, projLine))
                        projStreamlines.push_back(projLine);
                }
                parcelsRebuild();
                radiusChanged = false;
            }
            if (offsetPoly)
            {
                offsetPolys.clear();
                for (const auto &poly : cuttedPolys)
                {
                    std::vector<Polyline2_t<float>> offsets = geo::offsetPolygon(poly, offsetDist);
                    offsetPolys.insert(offsetPolys.end(), offsets.begin(), offsets.end());
                }
            }
        },
        [&]() { // 3维空间绘图内容部分
            for (size_t i = 0; i < draw_data.cellPolys.size(); i++)
            {
                render::stroke_light_polygon3(draw_data.cellPolys[i], RL_BLACK, 1.f, {terrain.getWidth() * terrain.getCellSize(), 0, 0});
                render::fill_polygon3(draw_data.cellPolys[i], room2colors[plan_prob.site2room[i]], 0.8f, {terrain.getWidth() * terrain.getCellSize(), 0, 0});
            }
            render::draw_points(draw_data.sites_world, RL_RED);

            for (size_t i = 0; i < draw_data.wall_edge_list.size(); i++)
            {
                render::draw_bold_polyline3(draw_data.wall_edge_list[i], RL_BLACK, 0.06f, 1.f, {terrain.getWidth() * terrain.getCellSize(), 0, 0});
            }

            // draw plsg voronoi
            // if(result.voronoi_cells.size()>0){
            //     for (const auto &p : result.voronoi_cells)
            //     {
            //         render::stroke_bold_polygon2(Polyloop2(p.points), RL_DARKPURPLE, 0.F, pathWidth, 1.F, {terrain.getWidth() * terrain.getCellSize(), 0});
            //     }
            // }
          

           

            render::draw_points(testParcelBounds, ptData.color, ptData.color.a, ptData.size, 0.f, {terrain.getWidth() * terrain.getCellSize(), 0});
            terrain.draw();
            terrain.drawContours(layers);
            if (mainPaths.size() > 0)
                for (const auto &path : mainPaths)
                    terrain.drawPath(path.path, pathWidth);
            // terrain.drawGraphEdges(debugCx,debugCy,debugRank);
            // DrawGrid(20,1.f);
            // DrawLine3D({0, 0, 0}, {10000, 0, 0}, RED);
            // DrawLine3D({0, 0, 0}, {0, 10000, 0}, BLUE);
            // DrawLine3D({0, 0, 0}, {0, 0, -10000}, GREEN);

            render::draw_points(seedPoints, RL_YELLOW, 0.25f, 1.2f);
            render::draw_points(roadControlPts, RL_GREEN, 0.25f, 0.6f);
            for (int i = 0; i < parcels.size(); i++)
            {
                // render::draw_points(parcels[i].points, ptData.color, ptData.color.a, ptData.size, 0.f, {terrain.getWidth() * terrain.getCellSize(), 0});
                render::draw_points(parcelObbs[i].points, ptData.color, ptData.color.a, ptData.size, 0.f, {terrain.getWidth() * terrain.getCellSize(), 0});
            }

            for (const auto &path : pathPolylines)
            {
                render::draw_bold_polyline2(path.points, RL_RED, 0.F, pathWidth, 1.f, {terrain.getWidth() * terrain.getCellSize(), 0});
            }
            for (const field::Tensor<float> &t : tensorField.getAllTensors())
            {
                for (int i = 0; i < 4; ++i)
                    render::draw_vector(t.pos, t.dirs[i], vecData.color, vecData.scale, vecData.startThickness, vecData.endThickness, vecZ, vecData.color.a);
            }
            // for (const auto &line : streamlines)
            // {
            //     render::draw_bold_polyline2(line.points, lineData.color, vecZ, lineData.Thickness, lineData.color.a);
            // }

            if (showFieldLine)
            {
                for (const auto &line : projStreamlines)
                {
                    render::draw_bold_polyline3(line, lineData.color, lineData.Thickness, lineData.color.a);
                }

                for (const auto &line : streamlines)
                {
                    render::draw_bold_polyline2(line.points, lineData.color, 0.f, lineData.Thickness, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
                }
            }

            for (const auto &attr : tensorField.getAttractors())
                attr.draw();

            for (int i = 0; i < projParcels.size(); i++)
            {
                Color c = renderUtil::room_color_from_id(i, projParcels.size());
                render::draw_bold_polyline3(projParcels[i], c, pathWidth, lineData.color.a);
            }

            if (/*parcels.size() > 0 && parcelObbs.size() > 0 && parcelhulls.size()>0&& parcelhullObbs.size()>0*/ true)
            {
                for (int i = 0; i < parcels.size(); i++)
                {
                    Color c = renderUtil::ColorFromHue((float)i / parcels.size());
                    render::draw_bold_polyline2(parcels[i].points, RL_BLACK, 0.f, pathWidth / 3.f, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
                    render::fill_polygon2(Polyloop2(parcels[i].points), c, 0.f, 0.3f, {terrain.getWidth() * terrain.getCellSize(), 0});
                    render::draw_bold_polyline2(parcelObbs[i].points, RL_RED, 0.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
                }
            }

            render::draw_bold_polyline2(unionPoly.points, RL_RED, 30.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});

            for (const auto &poly : subedPolys)
            {
                render::draw_bold_polyline2(poly.points, RL_DARKGREEN, 40.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
            }

            for (const auto &poly : interPolys)
            {
                render::draw_bold_polyline2(poly.points, RL_DARKBLUE, 50.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
            }

            for (const auto &poly : cuttedPolys)
            {
                render::draw_bold_polyline2(poly.points, RL_DARKPURPLE, 60.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
            }

            for (const auto &poly : offsetPolys)
            {
                render::draw_bold_polyline2(poly.points, RL_DARKGREEN, 60.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
            }

            // extrudeMesh.draw(color, colorAlpha,outline, wireframe, wireframeAlpha);
            // extrudeMesh_2.draw(GREEN, colorAlpha,outline, wireframe, wireframeAlpha);
            // extrudeMesh_3.draw(RED, colorAlpha,outline, wireframe, wireframeAlpha);
            // render::fill_polygon3(polyloop, RED, 0.5f);
            // render::stroke_bold_polygon3(polyloop, BLACK);
        },
        [&]() { // 二维屏幕空间绘图
            // render.draw_index_fonts(vertices, 16, BLUE);
            terrain.drawContourPtIndices(layers, render);
            if (showText)
            {
                for (const auto &parcel : parcels)
                {
                    render.draw_index_fonts(parcel.points, fontData.size, fontData.color, 0.f, {terrain.getWidth() * terrain.getCellSize(), 0});
                }
            }

            rlImGuiBegin(); // 开始ImGui帧渲染（必须在2D阶段调用）
            // bool demoOpen = true;
            // ImGui::ShowDemoWindow(&demoOpen);

            // 2. 自定义GUI窗口（纯2D固定在屏幕上）
            bool customOpen = true;
            if (ImGui::Begin("Render Settings", &customOpen))
            {
                if (ImGui::CollapsingHeader("Camera Control", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::SliderFloat("Camera Move Speed", &render::RENDER_MOVE_SPEED, 0.f, 2.f, "%.2f");
                    ImGui::SliderFloat("Camera Rotate Speed", &render::RENDER_ROTATE_SPEED, 0.f, 0.009f, "%.3f");
                    ImGui::SliderFloat("Camera Zoom Speed", &render::RENDER_ZOOM_SPEED, 0.f, 10.f, "%.2f");
                }

                if (ImGui::TreeNodeEx("Geo Settings", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::TreeNodeEx("Point Settings", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::SliderFloat("Point Size", &ptData.size, 0.01f, 10.f, "%.1f");
                        if (ImGui::ColorEdit4("Point Color", (float *)&ptData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                            ptData.syncFloatToColor();
                        ImGui::TreePop();
                    }
                    if (ImGui::TreeNodeEx("Vector Settings", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::SliderFloat("Vector Scale", &vecData.scale, 1.f, 20.f, "%.1f");
                        ImGui::SliderFloat("Start Thickness", &vecData.startThickness, 0.1f, 10.f, "%.1f");
                        ImGui::SliderFloat("End Thickness", &vecData.endThickness, 0.1f, 10.f, "%.1f");
                        ImGui::SliderFloat("Vec Z", &vecZ, -100.0f, 100.f, "%.1f");
                        if (ImGui::ColorEdit4("Vector Color", (float *)&vecData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                            vecData.syncFloatToColor();
                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNodeEx("Font Settings", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::SliderInt("Font Size", &fontData.size, 1, 30);
                        if (ImGui::ColorEdit4("Font Color", (float *)&fontData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                            fontData.syncFloatToColor();
                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNodeEx("Line Settings", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::SliderFloat("Line Thickness", &lineData.Thickness, 0.01f, 1.f, "%.2f");
                        if (ImGui::ColorEdit4("Line Color", (float *)&lineData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                            lineData.syncFloatToColor();
                        ImGui::TreePop();
                    }

                    ImGui::TreePop();
                }
            }
            ImGui::End();
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
                    if (ImGui::Button("Reset Terrain"))
                    {
                        terrain.regenerate((unsigned)time(nullptr), 1 << terrainPow, 1 << terrainPow, frequency, amplitude);
                        terrain.applyFaceColor();
                        adj = terrain.buildAdjacencyGraph(debugRank);
                        mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);
                        pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
                        //result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                        layers = terrain.extractContours(1.f);
                        tensorField = TensorField2D(terrain.getAABB2(), minGridNum);
                        terrainTensors = terrain.sampleTensorAtGrids(tensorField.getGridPoints());
                        tensorField.addConstraint(pathPolylines, attractors, terrainTensors);
                        samplePoints = M2::gen_poisson_sites_in_poly(terrainBounding2, 1.2f * tensorField.getGridSize(), 30, (unsigned)time(nullptr));
                        streamlines = tensorField.genStreamlines(samplePoints);
                        projStreamlines.clear();
                        for (const auto &l : streamlines)
                        {
                            std::vector<Vector3f> projLine;
                            if (terrain.projectPolylineToTerrain(l.points, projLine))
                                projStreamlines.push_back(projLine);
                        }
                        parcelsRebuild();
                    }
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

                // =====================================================
                // Graph Debug
                // =====================================================
                if (ImGui::CollapsingHeader("Road Path Graph Debug"))
                {
                    ImGui::Indent();
                    ImGui::InputInt("Center X", &debugCx);
                    ImGui::InputInt("Center Y", &debugCy);
                    weightChanged |= ImGui::InputInt("Rank", &debugRank);
                    ImGui::Separator();
                    if (ImGui::TreeNode("Cost Weights"))
                    {
                        weightChanged |= ImGui::SliderFloat("Dist Weight", &terrain.w_dist, 0.0f, 100.0f, "%.1f");
                        weightChanged |= ImGui::SliderFloat("Slope Weight", &terrain.w_slope, 0.0f, 100.0f, "%.1f");
                        weightChanged |= ImGui::SliderFloat("Up Weight", &terrain.w_up, 0.0f, 100.0f, "%.1f");
                        weightChanged |= ImGui::SliderFloat("Down Weight", &terrain.w_down, 0.0f, 100.0f, "%.1f");
                        ImGui::TreePop();
                    }
                    ImGui::Separator();
                    ImGui::SliderFloat("Path Width", &pathWidth, 0.05f, 1.0f, "%.2f");
                    ImGui::SliderInt("MainRoadNodeNum", &mainRoadNode, 0, 30);
                    debugCx = std::clamp(debugCx, 0, terrain.getWidth());
                    debugCy = std::clamp(debugCy, 0, terrain.getHeight());
                    debugRank = std::max(1, debugRank);

                    if (ImGui::Button("Debug Build Graph"))
                    {
                        terrain.drawGraphEdges(debugCx, debugCy, debugRank);
                        terrain.debugBuildGraphAt(debugCx, debugCy, debugRank);
                    }
                    ImGui::Unindent();
                }

                // =====================================================
                // Pathfinding
                // =====================================================
                // if (ImGui::CollapsingHeader("Pathfinding"))
                // {
                //     ImGui::Indent();
                //     ImGui::SliderInt("Start Vertex", &start, 0, static_cast<int>(terrain.getMesh().vertices.size() - 1));

                //     ImGui::SliderInt("Target Vertex", &target, 0, static_cast<int>(terrain.getMesh().vertices.size() - 1));

                //     ImGui::Unindent();
                // }

                // =====================================================
                // Terrain Scoring
                // =====================================================
                if (ImGui::CollapsingHeader("Terrain Scoring"))
                {
                    ImGui::Indent();
                    scoreWeightChanged |= ImGui::SliderFloat("Seed Aspect Weight", &terrain.wv_aspect, 0.f, 100.f, "%.1f");

                    scoreWeightChanged |= ImGui::SliderFloat("Seed Slope Weight", &terrain.wv_slope, 0.f, 100.f, "%.1f");

                    ImGui::Separator();
                    ImGui::SliderFloat("Score Threshold", &score_threshold, 0.f, 1.f, "%.2f");
                    ImGui::Unindent();
                }

                // =====================================================
                // Utilities
                // =====================================================
                if (ImGui::CollapsingHeader("Utilities"))
                {
                    ImGui::Indent();
                    ImGui::SliderFloat("Extrude Height", &extrudeHeight, 0.f, 50.f, "%.2f");
                    ImGui::ColorEdit3("Extrude Color", (float *)&color);
                    ImGui::SliderFloat("Extrude Alpha", &colorAlpha, 0.f, 1.f, "%.1f");
                    ImGui::Checkbox("Outline", &outline);
                    ImGui::Checkbox("Wireframe", &wireframe);
                    ImGui::SliderFloat("Wireframe Alpha", &wireframeAlpha, 0.f, 1.f, "%.1f");
                    ImGui::Separator();
                    ImGui::Checkbox("ShowFieldLines", &showFieldLine);
                    ImGui::Checkbox("ShowText", &showText);
                    genTensorfield |= ImGui::SliderInt("Tensor Div Nums", &minGridNum, 10, 60);
                    terrainfieldWeightChanged |= ImGui::SliderFloat("Terrain Field Weight", &terrainWeight, 1.f, 10.f, "%.1f");
                    radiusChanged |= ImGui::SliderFloat("Point Radius", &radius, 7.f, 30.f, "%.1f");
                    offsetPoly |= ImGui::SliderFloat("OffsetDist", &offsetDist, -10.f, 10.f, "%.1f");
                    ImGui::Unindent();
                }
            }
            ImGui::End();

            rlImGuiEnd();
        }});

    return 0;
}