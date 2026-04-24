#include <iostream>
#include "render.h"
#include "terrain.h"
#include "renderUtil.h"
#include <rlImGui.h>
#include <imgui.h>
#include "tensorField.h"
#include "optimizer.h"
#include "SCARoadGenerator.h"
#include "building.h"
#include "layout.h"
#include <fstream>
#include <vector>
#include <sstream>

using Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector2d;
using field::TensorField2D, field::PointAtrractor, field::TerrainTensor;
using render::Renderer3D;
using terrain::Terrain, terrain::TerrainCell, terrain::TerrainViewMode, terrain::ContourLayer, terrain::Road;
using namespace render;
using namespace optimizer;
using namespace geo;
using namespace SCARoad;
using namespace polyloop;
using namespace layout;
using namespace grid;

void loadDataFromFile(const std::string &filePath, std::vector<float> &heights, int width, int height)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    // 清空 heights 数组
    heights.clear();
    heights.resize(width * height); // 预先分配空间

    // 按列优先顺序读取数据并转换为行优先顺序
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float value;
            file >> value;

            // 将读取的值存储到 heights 数组中，按行优先存储
            heights[row * width + col] = value;
        }
    }

    file.close();
}

std::vector<Eigen::Vector3f> loadCoordinatesFromFile(const std::string &filePath)
{
    std::vector<Eigen::Vector3f> points; // 用于存储读取到的坐标点

    std::ifstream file(filePath); // 打开文件
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return points;
    }

    std::string line;
    while (std::getline(file, line)) // 逐行读取文件
    {
        // 去掉花括号
        line.erase(0, 1);            // 删除前面的 '{'
        line.erase(line.size() - 1); // 删除最后的 '}'

        // 使用字符串流分割坐标
        std::stringstream ss(line);
        float x, y, z;
        char comma;                              // 用于丢弃逗号
        if (ss >> x >> comma >> y >> comma >> z) // 读取 x, y, z 值
        {
            // 将读取到的坐标存储到 Eigen::Vector3f 中
            points.push_back(Eigen::Vector3f(x, y, z));
        }
    }

    file.close();  // 关闭文件
    return points; // 返回存储的坐标点
}

Polyline2_t<float> loadCoordinates2DFromFile(const std::string &filePath)
{
    std::vector<Eigen::Vector2f> points; // 用于存储读取到的坐标点

    std::ifstream file(filePath); // 打开文件
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return points;
    }

    std::string line;
    while (std::getline(file, line)) // 逐行读取文件
    {
        // 去掉花括号
        line.erase(0, 1);            // 删除前面的 '{'
        line.erase(line.size() - 1); // 删除最后的 '}'

        // 使用字符串流分割坐标
        std::stringstream ss(line);
        float x, y, z;
        char comma;                              // 用于丢弃逗号
        if (ss >> x >> comma >> y >> comma >> z) // 读取 x, y, z 值
        {
            // 将读取到的坐标存储到 Eigen::Vector3f 中
            points.push_back(Eigen::Vector2f(x, y));
        }
    }

    file.close(); // 关闭文件
    std::reverse(points.begin(), points.end());
    return Polyline2_t<float>(points, true);
}


void export3Dpts(std::vector<std::vector<Eigen::Vector3f>> polys, const std::string &outputDir){

    for (size_t i = 0; i < polys.size(); ++i)
    {
        
        std::string filename = outputDir + "/Road_"  + std::to_string(i) + ".txt";
        std::ofstream outFile(filename); // 打开文件进行写入

        if (!outFile.is_open())
        {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // 遍历当前 Polyline2_t 的所有点并写入文件
        const std::vector<Eigen::Vector3f> &p = polys[i];
        for (const auto &point : p)
        {

            outFile << point.x() << "," << point.y() << "," << point.z() << std::endl;
        }

        outFile.close(); // 关闭文件
        // std::cout << "Exported points to: " << filename << std::endl;
    }
    
}

int main()
{
    const std::string &outputDir  = "D:/Thesis/TerrainBuildingLayout/export/test1"; 
    std::vector<Eigen::Vector2f> riverPts = {
        {-205.918236f, -265.843369f},
        {-208.605342f, -206.957638f},
        {-188.519144f, -218.826754f},
        {-158.846353f, -228.413349f},
        {-120.043472f, -221.565781f},
        {-102.239797f, -203.305602f},
        {-107.717851f, -174.089315f},
        {-113.652409f, -150.807587f},
        {-107.261347f, -144.873028f},
        {-78.501564f, -173.176306f},
        {-55.676340f, -183.675909f},
        {-25.090540f, -171.350288f},
        {85.482123f, -106.013055f},
        {166.074889f, -54.163164f},
        {228.632911f, -5.694787f},
        {263.000000f, 10.934450f},
        {262.000000f, -24.832494f},
        {223.016169f, -52.249151f},
        {188.610997f, -95.289723f},
        {157.050194f, -129.104870f},
        {136.761106f, -182.645519f},
        {111.963331f, -198.989507f},
        {70.821569f, -210.261222f},
        {-0.905150f, -256.000000f},
        {-0.905150f, -266.000000f},
        {-205.918236f, -265.843369f}};

    std::vector<Eigen::Vector2f> constraintLinepts = {
        {-256.000000f, 15.402521f},
        {-90.512579f, 134.024161f},
        {-49.773474f, 215.502341f},
        {0.000000f, 231.797995f},
        {58.000000f, 199.000000f},
        {170.217618f, 256.241415f}};
    std::reverse(riverPts.begin(), riverPts.end());
    std::cout << "riverPts size: " << riverPts.size() << "\n"; // 检查点数据是否有效
    Polyline2_t<float> river(riverPts, true);
    std::cout << "River points size: " << river.points.size() << "\n"; // 检查 river 对象中的点是否正确
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

    static int terrainPow = 7;
    static float frequency = 0.02f;
    static float amplitude = 16.5f;
    static int mainRoadNode = 16;
    int lastMainRoadNode = -1;
    static float extrudeHeight = 2.f;
    float lastExtrudeHeight = -1.f;
    static float colorAlpha = 0.5f;
    static float wireframeAlpha = 0.8f;
    bool wireframe = true;
    bool outline = true;

    static int secRoadIterstep = 500;
    PointDrawData ptData;
    VectorDrawData vecData;
    FontDrawData fontData;
    LineDrawData lineData;
    static float vecZ = 0.f;
    static int minGridNum = 30;
    static float terrainWeight = 2.f;
    bool genTensorfield = false;
    bool terrainfieldWeightChanged = false;
    bool showFieldLine = false;
    bool showText = false;
    static bool showTerrain = false;
    static bool showProjectedRoads = false;
    static bool showPlanarRoads = false;
    static bool showPlanarSites = false;
    Eigen::Vector2i roadGapRange = {18, 55};
    Eigen::Vector2i depthRange = {18, 55};

    // Input HeightMap to generate terrain
    std::vector<float> heightMap;
    std::string filePath = R"(D:\Thesis\TerrainBuildingLayout\heightMap.txt)";
    std::string river3DfilePath = R"(D:\Thesis\TerrainBuildingLayout\riverPoints.txt)";
    std::string boundFilePath = R"(D:\Thesis\TerrainBuildingLayout\bound.txt)";
    // std::cout << "File path: " << filePath << std::endl;
    loadDataFromFile(filePath, heightMap, 257, 257);
    // Terrain terrain((unsigned)time(nullptr), 1 << terrainPow, 1 << terrainPow, 2.f, frequency, amplitude);

    Terrain terrain(heightMap, 256, 256, 2.f);
    std::cout << "height map size is " << heightMap.size() << "\n";
    std::vector<Eigen::Vector3f> river3Dpts = loadCoordinatesFromFile(river3DfilePath);
    Polyline2_t<float> areaBound = loadCoordinates2DFromFile(boundFilePath);
    std::cout << "Printing points of areaBound:" << std::endl;
    for (size_t i = 0; i < areaBound.points.size(); ++i)
    {
        const auto &p = areaBound.points[i];
        std::cout << "Point " << i << ": (" << p.x() << ", " << p.y() << ")" << std::endl;
    }

    TerrainViewMode viewMode = TerrainViewMode::Lit;
    terrain.setViewMode(viewMode);
    std::vector<ContourLayer> layers = terrain.extractContours(1.f);

    std::vector<std::vector<geo::GraphEdge>> adj = terrain.buildAdjacencyGraph(debugRank);
    // std::vector<int> path;
    bool needGenTerrain = false;
    bool weightChanged = false;
    bool scoreWeightChanged = false;

    std::vector<Eigen::Vector3f> seedPoints;
    std::vector<Eigen::Vector3f> roadControlPts;
    static float radius = 150.f;
    bool radiusChanged = false;
    std::vector<PointAtrractor<float>> attractors;
    attractors.emplace_back(Vector2f(153.f, -208.f), radius);
    TensorField2D<float> tensorField = TensorField2D(terrain.getAABB2(), minGridNum);
    std::unordered_map<int, field::TerrainTensor<float>> terrainTensors = terrain.sampleTensorAtGrids(tensorField.getGridPoints());
    std::vector<Road> mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);

    std::vector<Polyline2_t<float>> pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
    std::vector<Polyline2_t<float>> rivers = {river};
    tensorField.addConstraint({river, Polyline2_t<float>(constraintLinepts)}, attractors, terrainTensors);
    bool isInside = util::Math2<float>::point_in_poly(river.points, {-100.f, -256.f});
    std::cout << "test point is inside river " << isInside << "\n";
    int testA_winding = util::Math2<float>::winding_number(river.points, {-100.f, -256.f});
    int testB_winding = util::Math2<float>::winding_number(river.points, {-300.f, -256.f});
    std::cout << "testA_winding is " << testA_winding << "\n";
    std::cout << "testB_winding is " << testB_winding << "\n";
    std::vector<std::vector<int>> mainPathIndices;
    for (const auto &path : mainPaths)
    {
        mainPathIndices.push_back({path.path.begin(), path.path.end()});
    }

    static int terrain_width = terrain.getCellSize() * terrain.getWidth();
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

    Polyline2_t<float> testPolyA({{0.f, 0.f}, {50.f, 50.f}, {100.f, 0.f}, {100.f, 100.f}, {0.f, 100.f}, {0.f, 0.f}}, true);
    Polyline2_t<float> testPolyB({{10.f, 0.f}, {31.f, 11.f}, {12.f, 14.f}, {10.f, 0.f}}, true);
    std::vector<Vector2f> cutPoints = {{0.f, 10.f}, {20.f, 10.f}, {10.f, 30.f}, {-5.f, 10.f}};
    // cutPoints.insert(cutPoints.end(), std::make_move_iterator(cutPoints.end()), std::make_move_iterator(cutPoints.begin()));
    Polyline2_t<float> cutline(cutPoints);

    Polyline2_t<float> unionPoly = geo::unionPolygon(testPolyA, testPolyB);
    // std::vector<Polyline2_t<float>> subedPolys = geo::subPolygon(testPolyA, testPolyB);
    // std::vector<Polyline2_t<float>> interPolys = geo::intersectPolygon(testPolyA, testPolyB);
    // std::vector<Polyline2_t<float>> cuttedPolys = geo::splitPolygonByPolylines(testPolyA, {cutline});
    // std::cout << "cuttedpolys is " << cuttedPolys.size() << std::endl;
    std::vector<Polyline2_t<float>> offsetPolys;
    static float offsetDist = -2.5f;
    static float buildingDepth = -15.f;
    static float offsetDist2 = -1.2f;
    // std::vector<Polyline2_t<float>> divPolys = geo::generateRandomPolysAlongPolygon(testPolyA, offsetDist, 7.f, 12.f);
    std::vector<Polyline2_t<float>> weightOffsetPoly = geo::offsetPolygon(testPolyA, offsetDist, {1.f, 1.f, 0.5f, 0.07f, 0.5f});
    Polyline2_t<float> averageOffsetPoly = geo::offsetPolygon(testPolyA, offsetDist)[0];
    bool offsetPoly = false;

    bool offsetInside = false;
    // for (const auto &poly : divPolys)
    // {
    //     std::vector<Polyline2_t<float>> offsets = geo::offsetPolygon(poly, offsetDist2);
    //     offsetPolys.insert(offsetPolys.end(), offsets.begin(), offsets.end());
    // }

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
    geo::PolygonMesh extrudeMesh = geo::PolygonMesh({{0, 0, 0}, {30, 0, 0}, {30, 15, 0}, {45, 15, 0}, {45, 30, 0}, {0, 30, 0}}, extrudeHeight);
    // geo::PolygonMesh extrudeMesh_2 = geo::PolygonMesh({{0, 0, extrudeHeight}, {30, 0, extrudeHeight}, {30, 30, extrudeHeight}, {0, 30, extrudeHeight}}, extrudeHeight);
    // geo::PolygonMesh extrudeMesh_3 = geo::PolygonMesh({{30, 0, 0}, {45, 0, 0}, {45, 15, 0}, {30, 15, 0}}, extrudeHeight);
    // polyloop::Polyloop3 polyloop({{0, 0,0}, {30, 0,0}, {30, 30,0}, {0, 30,0}});
    Color color = RL_YELLOW;

    // ============================
    // 1. 初始化 seeds
    SCANode root;
    root.position = Eigen::Vector2f(0, 0);
    std::vector<SCANode> scaNodes;
    scaNodes.push_back(root);

    // ============================
    // 2. 初始化 attractors
    std::vector<Eigen::Vector2f> attractPos;
    std::vector<SCARoad::Attractor> scaAttractors = SCARoad::getRandomAttractors(35.f, 512, 512, attractPos);

    SCANetwork net(scaNodes, scaAttractors, terrain, tensorField);

    std::vector<Eigen::Vector2f> nodePoints;
    for (auto &n : net.nodes)
        nodePoints.push_back(n.position);
    std::cout << "Init nodes: " << net.nodes.size() << "\n";
    std::cout << "Init attractors: " << net.attractors.size() << "\n";

    // ============================
    // 3. 迭代
    // ============================
    int maxSCAIter = 800;
    auto scaRoads = net.extractRoads();
    std::vector<std::vector<Vector3f>> projSCAroads;
    std::vector<Polyline2_t<float>> realSites;
    std::vector<Polyline2_t<float>> allOffsetSites;
    std::vector<Polyline2_t<float>> insideBoundSites;
    for (const auto &l : scaRoads)
    {
        std::vector<Vector3f> projLine;
        if (terrain.projectPolylineToTerrain(l, projLine))
            projSCAroads.push_back(projLine);
    }

    int scaIter = 0;
    bool scaUpdated = false;

    bool growthStopped = false;
    bool showAttrIndices = true;

    bool showVolumes = true;
    bool showFloorVolumes = false;
    bool applyRiverColor = false;
    bool applyYardColor = false;
    bool showYardVolumes = false;
    bool showAtrractor = false;
    static float OutlineThickness = 0.07f;
    std::vector<building::Building> buildings;

    std::vector<building::Volume> volumes;
    static int reBuildIndex = 0;
    std::vector<geo::Polyline2_t<float>> yardPolys;
    std::vector<geo::Polyline2_t<float>> rectSites;
    Ray ray = {0};
    rlImGuiSetup(true);

    std::vector<Eigen::Vector3f> volumeCenters;

    auto terrainRebuild = [&]() {

    };

    auto tensorFieldRebuild = [&]
    {
        tensorField.addConstraint({river, Polyline2_t<float>(constraintLinepts)}, attractors, terrainTensors);
        samplePoints = M2::gen_poisson_sites_in_poly(terrainBounding2, 1.2f * tensorField.getGridSize(), 30, (unsigned)time(nullptr));
        streamlines = tensorField.genStreamlines(samplePoints);
        projStreamlines.clear();

        for (const auto &l : streamlines)
        {
            std::vector<Vector3f> projLine;
            if (terrain.projectPolylineToTerrain(l.points, projLine))
                projStreamlines.push_back(projLine);
        }
        mainPathIndices.clear();
        for (const auto &path : mainPaths)
        {
            mainPathIndices.push_back({path.path.begin(), path.path.end()});
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
            if (IsKeyPressed(KEY_S))
            {

                scaUpdated = true;
                scaIter = 0;
                scaRoads.clear();
                realSites.clear();
                offsetPolys.clear();
                insideBoundSites.clear();
                volumes.clear();
                scaAttractors = SCARoad::getRandomAttractors(35.f, 512, 512, attractPos);
                net.rebuildNet(scaNodes, scaAttractors);
                if (growthStopped)
                    scaUpdated = false;
                scaIter++;

                // net.update(growthStopped);
                // nodePoints.clear();
                // for (auto &n : net.nodes)
                //     nodePoints.push_back(n.position);
                // scaRoads = net.extractRoads();
                // projSCAroads.clear();
                // for (const auto &l : scaRoads)
                // {
                //     std::vector<Vector3f> projLine;
                //     if (terrain.projectPolylineToTerrain(l, projLine))
                //         projSCAroads.push_back(projLine);
                // }
                // std::cout << "[Iter " << scaIter << "] "
                //           << "nodes=" << net.nodes.size()
                //           << " attractors=" << net.attractors.size()
                //           << "\n";

                // if (net.attractors.empty())
                // {
                //     scaUpdated = false;
                //     std::cout << "All attractors consumed.\n";
                // }
            }

            if (scaUpdated)
            {
                if (scaIter < maxSCAIter)
                {
                    scaIter++;
                    net.update(growthStopped);
                    nodePoints.clear();
                    for (auto &n : net.nodes)
                        nodePoints.push_back(n.position);
                    scaRoads = net.extractRoads();
                    // std::cout << "[Iter " << scaIter << "] "
                    //           << "nodes=" << net.nodes.size()
                    //           << " attractors=" << net.attractors.size()
                    //           << "\n";
                    if (growthStopped)
                        scaUpdated = false;
                    if (net.attractors.empty())
                    {
                        scaUpdated = false;
                        std::cout << "All attractors consumed.\n";
                    }
                }
                else
                    scaUpdated = false;
            }
            if (IsKeyPressed(KEY_C))
            {
                net.finalConnectNodes(10, 17, 6);
                scaRoads = net.extractRoads();
                projSCAroads.clear();
                for (const auto &l : scaRoads)
                {
                    std::vector<Vector3f> projLine;
                    if (terrain.projectPolylineToTerrain(l, projLine))
                        projSCAroads.push_back(projLine);
                }
            }

            if (IsKeyPressed(KEY_P))
            {
                scaRoads.clear();
                projSCAroads.clear();
                realSites.clear();
                allOffsetSites.clear();
                buildings.clear();
                offsetPolys.clear();
                insideBoundSites.clear();
                volumes.clear();
                const auto &allRoads = net.extractCloseAndLinearRoads();
                for (const auto &poly : allRoads.first)
                {
                    bool isAllInside = true;
                    bool isAllOutside = true;

                    // std::cout << "Processing polygon with " << poly.points.size() << " points." << std::endl;

                    // 遍历 poly 的每个点，判断哪些点在 areaBound 内
                    for (const auto &p : poly.points)
                    {
                        if (util::Math2<float>::point_in_poly(areaBound.points, p))
                        {
                            isAllOutside = false; // 如果某点在区域内，标记 isAllOutside 为 false
                            // std::cout << "Point (" << p.x() << ", " << p.y() << ") is inside the area." << std::endl;
                        }
                        else
                        {
                            isAllInside = false; // 如果某点在区域外，标记 isAllInside 为 false
                            // std::cout << "Point (" << p.x() << ", " << p.y() << ") is outside the area." << std::endl;
                        }
                    }

                    if (isAllInside)
                    {
                        std::cout << "Polygon is entirely inside the area." << std::endl;
                        insideBoundSites.push_back(poly);
                    }

                    // 如果 poly 既有点在区域内又有点在区域外，就进行减法操作
                    if (!isAllInside && !isAllOutside)
                    {
                        std::cout << "Polygon intersects the area, performing intersection operation." << std::endl;
                        std::vector<Polyline2_t<float>> result = geo::intersectPolygon(poly, areaBound);

                        // 处理交集部分
                        for (const auto &r : result)
                        {
                            if (util::Math2<float>::polygon_area(r.points) < 150.f)
                            {
                                // std::cout << "Skipping small polygon with area less than threshold (150)." << std::endl;
                                continue;
                            }
                            // std::cout << "Adding intersection polygon with area: "
                            //           << util::Math2<float>::polygon_area(r.points) << std::endl;
                            insideBoundSites.push_back(r);
                        }
                    }

                    // 输出处理完成的 polygon 信息
                    // std::cout << "Finished processing polygon." << std::endl;
                }
                for (const auto &poly : insideBoundSites)
                {
                    scaRoads.push_back(poly.points);
                    std::vector<Polyline2_t<float>> offsetSites = geo::offsetPolygon(poly, offsetDist);
                    realSites.insert(realSites.end(), offsetSites.begin(), offsetSites.end());
                    for (const auto &poly : offsetSites)
                    {
                        std::vector<Polyline2_t<float>> siteOffsets = geo::offsetPolygon(poly, buildingDepth);
                        allOffsetSites.insert(allOffsetSites.end(), siteOffsets.begin(), siteOffsets.end());
                        std::vector<Polyline2_t<float>> divSitePolys = geo::generateRandomPolysAlongPolygon(poly, buildingDepth, 15.f, 18.f);
                        // divPolys.insert(divPolys.end(), divSitePolys.begin(), divSitePolys.end());
                        // offsetPolys.insert(offsetPolys.end(), divSitePolys.begin(), divSitePolys.end());
                        for (const auto &div : divSitePolys)
                        {
                            std::vector<Polyline2_t<float>> offsets = geo::offsetPolygon(div, offsetDist2);
                            // offsetPolys.insert(offsetPolys.end(), offsets.begin(), offsets.end());
                            // if (buildings.size() < 30)
                            // {
                            for (const auto &offset : offsets)
                            {
                                // --- 面积检查（推荐）
                                float area = M2::polygon_area(offset.points); // 你应该有这个函数
                                if (std::abs(area) < 120.f)
                                {
                                    // std::cout << "[WARNING] offset polygon too small\n";
                                    continue;
                                }

                                // --- 顶点数检查
                                if (offset.points.size() < 3)
                                {
                                    // std::cout << "[WARNING] invalid offset polygon\n";
                                    continue;
                                }

                                geo::OBB2<float> obb(offset.points);
                                Eigen::Matrix<float, 2, 2> R = geo::rotationToXAxis(obb.axis0);

                                Polyline2_t<float> rotPoly = geo::rotatePoly(offset, R);
                                Polyline2_t<float> rotedRect = geo::getMaxRectInPolyWithRatio(rotPoly, 1, 1.8);
                                 float rectArea = M2::polygon_area(rotedRect.points);
                                if (std::abs(rectArea) < 90.f)
                                {
                                    // std::cout << "[WARNING] offset polygon too small\n";
                                    continue;
                                }

                                // =========================
                                // 5. 构建 Building（重点保护）
                                // =========================
                                try
                                {
                                    // building::Building building(offset, terrain);

                                    // 可选：检查 mesh 是否有效
                                    // if (!building.buildingMesh.isValid()) continue;
                                    offsetPolys.push_back(offset);
                                    // volumes.emplace_back(offset, terrain);
                                    // buildings.push_back(std::move(building));
                                }
                                catch (const std::exception &e)
                                {
                                    std::cout << "[ERROR] Building construction failed: "
                                              << e.what() << "\n";
                                }
                                catch (...)
                                {
                                    std::cout << "[ERROR] Building construction crashed\n";
                                }
                            }
                            // }
                        }
                    }
                }
                // std::cout << "scaRoads size: " << scaRoads.size() << std::endl;
                for (const auto &poly : allRoads.second)
                {
                    std::vector<Eigen::Vector2<float>> newPoints;
                    bool isFirstPointInside = util::Math2<float>::point_in_poly(areaBound.points, poly.points[0]);

                    // 如果第一个点就不在区域内，跳过这条多边形
                    if (!isFirstPointInside)
                    {
                        continue;
                    }

                    bool wasInside = false;

                    // 遍历 poly.points 中的点
                    for (size_t i = 0; i < poly.points.size(); ++i)
                    {
                        const auto &p = poly.points[i];
                        bool pointInBound = util::Math2<float>::point_in_poly(areaBound.points, p);

                        if (pointInBound)
                        {
                            // 当前点在 areaBound 内，保留此点
                            newPoints.push_back(p);
                            wasInside = true;
                        }
                        else
                        {
                            // 当前点在 areaBound 外，且前一个点在区域内，删除该点之后的所有点
                            if (wasInside)
                            {
                                // 如果前一个点在区域内，且当前点在区域外，结束此条多边形的处理
                                break;
                            }
                        }
                    }

                    // 如果新的点集非空，加入到 scaRoads 中
                    if (!newPoints.empty())
                    {
                        scaRoads.push_back(newPoints);
                    }
                }
                for (const auto &l : scaRoads)
                {
                    std::vector<Vector3f> projLine;
                    if (terrain.projectPolylineToTerrain(l, projLine))
                        projSCAroads.push_back(projLine);
                }
            }

            if (IsKeyPressed(KEY_B))
            {
                volumes.clear();
                volumeCenters.clear();
                for (const auto &p : offsetPolys)
                {

                    const building::Volume &v = building::Volume(p, terrain);
                    volumes.push_back(v);

                    yardPolys.insert(yardPolys.end(), v.yardBounds.begin(), v.yardBounds.end());
                    const Polyline2_t<float> &oriRect = v.layout.realSite;
                    const Eigen::Vector2f &cen = util::Math2<float>::getPolygonCentroid(oriRect.points);
                    volumeCenters.push_back({cen.x(), cen.y(), v.maxHeight + 4.f});
                }
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
                mainPathIndices.clear();
                for (const auto &path : mainPaths)
                {
                    mainPathIndices.push_back({path.path.begin(), path.path.end()});
                }

                // result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                weightChanged = false;
            }
            if (scoreWeightChanged && terrain.getViewMode() == TerrainViewMode::Score)
            {
                terrain.calculateInfos();
                terrain.applyFaceColor();
                mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);
                pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
                mainPathIndices.clear();
                for (const auto &path : mainPaths)
                {
                    mainPathIndices.push_back({path.path.begin(), path.path.end()});
                }

                // result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                //  std::cout << "polylines num is " << pathPolylines.size() << std::endl;
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
                // result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                mainPathIndices.clear();
                for (const auto &path : mainPaths)
                {
                    mainPathIndices.push_back({path.path.begin(), path.path.end()});
                }

                tensorFieldRebuild();
                parcelsRebuild();
            }

            if (mainRoadNode != lastMainRoadNode)
            {
                lastMainRoadNode = mainRoadNode;
                mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);
                pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
                // result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
                mainPathIndices.clear();
                for (const auto &path : mainPaths)
                {
                    mainPathIndices.push_back({path.path.begin(), path.path.end()});
                }

                tensorFieldRebuild();
                parcelsRebuild();
            }
            if (needGenTerrain)
            {
                terrain.regenerate(1 << terrainPow, 1 << terrainPow, frequency, amplitude);
                adj = terrain.buildAdjacencyGraph(debugRank);
                mainPaths = terrain.buildRoads(seedPoints, roadControlPts, terrain.regionInfos, mainRoadNode, adj);
                pathPolylines = terrain.convertRoadToFieldLine(mainPaths);
                mainPathIndices.clear();
                for (const auto &path : mainPaths)
                {
                    mainPathIndices.push_back({path.path.begin(), path.path.end()});
                }

                // result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
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
                mainPathIndices.clear();
                for (const auto &path : mainPaths)
                {
                    mainPathIndices.push_back({path.path.begin(), path.path.end()});
                }

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
                mainPathIndices.clear();
                for (const auto &path : mainPaths)
                {
                    mainPathIndices.push_back({path.path.begin(), path.path.end()});
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
                mainPathIndices.clear();
                for (const auto &path : mainPaths)
                {
                    mainPathIndices.push_back({path.path.begin(), path.path.end()});
                }

                parcelsRebuild();
                radiusChanged = false;
            }
            if (offsetPoly)
            {

                realSites.clear();
                offsetPolys.clear();
                for (const auto &site : scaRoads)
                {
                    std::vector<Polyline2_t<float>> offsetSites = geo::offsetPolygon(Polyline2_t(site), offsetDist);
                    realSites.insert(realSites.end(), offsetSites.begin(), offsetSites.end());
                    for (const auto &poly : offsetSites)
                    {
                        std::vector<Polyline2_t<float>> divSitePolys = geo::generateRandomPolysAlongPolygon(poly, buildingDepth, 8.f, 10.f);
                        for (const auto &div : divSitePolys)
                        {
                            std::vector<Polyline2_t<float>> offsets = geo::offsetPolygon(div, offsetDist2);
                            offsetPolys.insert(offsetPolys.end(), offsets.begin(), offsets.end());
                        }
                    }
                }

                offsetPoly = false;
            }
            if (offsetInside)
            {
                offsetPolys.clear();
                for (const auto &poly : realSites)
                {
                    std::vector<Polyline2_t<float>> divSitePolys = geo::generateRandomPolysAlongPolygon(poly, buildingDepth, 8.f, 10.f);
                    // divPolys.insert(divPolys.end(), divSitePolys.begin(), divSitePolys.end());
                    for (const auto &div : divSitePolys)
                    {
                        std::vector<Polyline2_t<float>> offsets = geo::offsetPolygon(div, offsetDist2);
                        offsetPolys.insert(offsetPolys.end(), offsets.begin(), offsets.end());
                    }

                    // std::vector<Polyline2_t<float>> offsets = geo::offsetPolygon(Polyline2_t(poly), offsetDist);
                    // realSites.insert(realSites.end(), offsets.begin(), offsets.end());
                }
                offsetInside = false;
            }

        },
        [&]() { // 3维空间绘图内容部分
            // for (size_t i = 0; i < draw_data.cellPolys.size(); i++)
            // {
            //     render::stroke_light_polygon3(draw_data.cellPolys[i], RL_BLACK, 1.f, {terrain.getWidth() * terrain.getCellSize(), 0, 0});
            //     render::fill_polygon3(draw_data.cellPolys[i], room2colors[plan_prob.site2room[i]], 0.8f, {terrain.getWidth() * terrain.getCellSize(), 0, 0});
            // }
            // render::draw_points(draw_data.sites_world, RL_RED);

            // for (size_t i = 0; i < draw_data.wall_edge_list.size(); i++)
            // {
            //     render::draw_bold_polyline3(draw_data.wall_edge_list[i], RL_BLACK, 0.06f, 1.f, {terrain.getWidth() * terrain.getCellSize(), 0, 0});
            // }

            // draw plsg voronoi
            // if(result.voronoi_cells.size()>0){
            //     for (const auto &p : result.voronoi_cells)
            //     {
            //         render::stroke_bold_polygon2(Polyloop2(p.points), RL_DARKPURPLE, 0.F, pathWidth, 1.F, {terrain.getWidth() * terrain.getCellSize(), 0});
            //     }
            // }

            // for (int i = 0; i < rectSites.size(); i++)
            // {
            //     // Color c = renderUtil::ColorFromHue((float)i / realSites.size());
            //     Polyline2_t<float> site = rectSites[i];
            //     render::draw_bold_polyline2(site.points, RL_DARKPURPLE, 0.f, 0.07f, lineData.color.a /*,{terrain.getWidth() * terrain.getCellSize(), 0}*/);
            //     // render::fill_polygon2(Polyloop2(site.points), c, 0.f, 0.3f);
            // }

            // for (int i = 0; i < insideBoundSites.size(); i++)
            // {
            //     Color c = renderUtil::ColorFromHue((float)i / insideBoundSites.size());
            //     Polyline2_t<float> site = insideBoundSites[i];
            //     render::draw_bold_polyline2(site.points, RL_DARKPURPLE, 0.f, 0.07f, lineData.color.a /*,{terrain.getWidth() * terrain.getCellSize(), 0}*/);
            //     render::fill_polygon2(Polyloop2(site.points), c, 0.f, 0.3f);
            // }
            // extrudeMesh.draw(RL_LIGHTGRAY, 0.8f, true, false, 0.2f);
            // for (const auto &site : allOffsetSites)
            // {
            //     // Color c = renderUtil::ColorFromHue((float)i / realSites.size());
            //     // Polyline2_t<float> site = allOffsetSites[i];
            //     render::draw_bold_polyline2(site.points, RL_DARKPURPLE, 0.f, lineData.Thickness, lineData.color.a /*,{terrain.getWidth() * terrain.getCellSize(), 0}*/);
            //     // render::fill_polygon2(Polyloop2(site.points), c, 0.f, 0.3f);
            // }
            // if (showVolumes)
            // {
            // for (const auto &building : buildings)
            // {
            //     try
            //     {
            //         // building.buildingMesh.draw(RL_LIGHTGRAY, 0.6f, true, false, 0.2f);
            //         // for (const auto &mesh : floorSystem.floorMeshes)
            //         // {
            //         //     mesh.draw({200, 210, 210, 255}, 0.45f, true, false, 0.5f, {0.f, 0.f, 0.f}, RL_BLACK, 0.02f, 1.f);
            //         //     // mesh.draw(RL_GRAY, 0.6f, true, false, 0.5f, {6 * rectBoundSize.x(), 0.f, 0.f});
            //         // }
            //         // for (const auto &mesh : floorSystem.yardMeshes)
            //         // {
            //         //     mesh.draw({175, 212, 120, 255}, 0.6f, true, false, 0.5f, {0.f, 0.f, 0.f}, RL_BLACK, 0.02f, 0.5f);
            //         //     // mesh.draw({175, 212, 120, 255}, 0.6f, true, false, 0.5f, {6 * rectBoundSize.x(), 0.f, 0.f});
            //         // }
            //     }
            //     catch (...)
            //     {

            //     }
            // }
            // }

            render::stroke_bold_polygon3(river3Dpts, RL_RED, render.lineData.Thickness);
            if (showVolumes)
            {
                for (const auto &volume : volumes)
                {
                    try
                    {

                        for (const auto &mesh : volume.volumeMeshes)
                        {
                            mesh.draw({200, 210, 210, 255}, 0.85f, true, false, 0.5f, {0.f, 0.f, 0.f}, RL_BLACK, OutlineThickness, 1.f);
                            // mesh.draw(RL_GRAY, 0.6f, true, false, 0.5f, {6 * rectBoundSize.x(), 0.f, 0.f});
                        }
                    }
                    catch (...)
                    {
                    }
                }
            }

            if (showYardVolumes)
            {
                for (const auto &volume : volumes)
                {
                    try
                    {
                        for (const auto &mesh : volume.yardMeshes)
                        {
                            mesh.draw({175, 212, 120, 255}, 0.8f, true, false, 0.5f, {0.f, 0.f, 0.f}, RL_BLACK, OutlineThickness, 0.5f);
                            // mesh.draw({175, 212, 120, 255}, 0.6f, true, false, 0.5f, {6 * rectBoundSize.x(), 0.f, 0.f});
                        }
                    }
                    catch (...)
                    {
                    }
                }
            }

            if (showText)
            {
                render::draw_points(volumeCenters, ptData.color, ptData.color.a, ptData.size);
            }
            if (showTerrain)
                terrain.draw();
            terrain.drawContours(layers);
            if (showProjectedRoads)
            {
                for (const auto &l : projSCAroads)
                {
                    render::draw_bold_polyline3(l, lineData.color, pathWidth, lineData.color.a);
                }
            }

            if (showPlanarRoads)
            {
                for (const auto &road : scaRoads)
                {
                    render::draw_bold_polyline2(road, render.lineData.color, 0.f, render.lineData.Thickness, render.lineData.color.a);
                    // render::draw_points(road, RL_GREEN, 1.f, render.ptData.size / 2.f);
                }
                
            }

            if (showPlanarSites){
                for (int i = 0; i < offsetPolys.size(); i++)
                {
                    Color c = renderUtil::ColorFromHue((float)i / offsetPolys.size());
                    const auto &poly = offsetPolys[i];
                    render::draw_bold_polyline2(poly.points, RL_DARKPURPLE, 0.f, lineData.Thickness, lineData.color.a /*,{terrain.getWidth() * terrain.getCellSize(), 0}*/);
                    render::fill_polygon2(poly.points, c, 0.f, 0.3f);
                    render::draw_points(poly.points, render.ptData.color, 1.f, render.ptData.size);
                }
            }

                // for (const auto &volume : volumes)
                // {
                //     // render::stroke_bold_polygon2(volume.layout.realSite.points, RL_DARKPURPLE, 0.f, lineData.Thickness, lineData.color.a);
                //     // volume.showData.draw(4.f, 1.f);
                //     // volume.softModel.drawGrids(0.f, 1.f);
                //     render::draw_points(volume.layout.sampleCenters, render.ptData.color, 1.f, render.ptData.size);
                //     render::draw_bold_polyline2(volume.rectSite.points, RL_GREEN, 0.f, lineData.Thickness, lineData.color.a);
                //     render::draw_bold_polyline2(volume.rotPoly.points, RL_RED, 0.f, lineData.Thickness, lineData.color.a);
                //     render::draw_bold_polyline2(volume.temp.points, RL_YELLOW, 0.f, lineData.Thickness, lineData.color.a);
                //     render::stroke_bold_polygon2(volume.layout.rotedSite.points, RL_RED, 0.f, 0.07f, 1.f /* , {0, -4 * rectBoundSize.y()} */);
                //     // render::stroke_bold_polygon2(volume.layout.realSite.points, RL_BLUE, 0.f, 0.07f, 1.f);
                // }

                // if (mainPaths.size() > 0)
                //     for (const auto &path : mainPaths)
                //         terrain.drawPath(path.path, pathWidth);
                // terrain.drawGraphEdges(debugCx,debugCy,debugRank);
                // DrawGrid(20,1.f);
                // DrawLine3D({0, 0, 0}, {10000, 0, 0}, RL_RED);
                // DrawLine3D({0, 0, 0}, {0, 10000, 0}, RL_BLUE);
                // DrawLine3D({0, 0, 0}, {0, 0, -10000}, RL_GREEN);

                //----------------------------------------road path point debug----------------------------------------
                // render::draw_points(seedPoints, RL_YELLOW, 0.25f, 1.2f);
                // render::draw_points(roadControlPts, RL_GREEN, 0.25f, 0.6f);
                // for (int i = 0; i < parcels.size(); i++)
                // {
                //     // render::draw_points(parcels[i].points, ptData.color, ptData.color.a, ptData.size, 0.f, {terrain.getWidth() * terrain.getCellSize(), 0});
                //     render::draw_points(parcelObbs[i].points, ptData.color, ptData.color.a, ptData.size, 0.f, {terrain.getWidth() * terrain.getCellSize(), 0});
                // }

                // for (const auto &path : pathPolylines)
                // {
                //     render::draw_bold_polyline2(path.points, RL_RED, 0.F, pathWidth, 1.f, {terrain.getWidth() * terrain.getCellSize(), 0});
                // }
                // for (const field::Tensor<float> &t : tensorField.getAllTensors())
                // {
                //     for (int i = 0; i < 4; ++i)
                //         render::draw_vector(t.pos, t.dirs[i], vecData.color, vecData.scale, vecData.startThickness, vecData.endThickness, vecZ, vecData.color.a);
                // }
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

                    // for (const auto &line : streamlines)
                    // {
                    //     render::draw_bold_polyline2(line.points, Fade(RL_GRAY, 0.5f), 0.f, lineData.Thickness, lineData.color.a /* ,{terrain.getWidth() * terrain.getCellSize(), 0}*/);
                    // }
                }

            render::draw_points(attractPos, render.ptData.color, 1.f, render.ptData.size);

            //----------------------------attractor debug--------------------------------
            // for (const auto &attr : tensorField.getAttractors())
            //     attr.draw();

            // for (int i = 0; i < projParcels.size(); i++)
            // {
            //     Color c = renderUtil::room_color_from_id(i, projParcels.size());
            //     render::draw_bold_polyline3(projParcels[i], c, pathWidth, lineData.color.a);
            // }

            // if (/*parcels.size() > 0 && parcelObbs.size() > 0 && parcelhulls.size()>0&& parcelhullObbs.size()>0*/ true)
            // {
            //     for (int i = 0; i < parcels.size(); i++)
            //     {
            //         Color c = renderUtil::ColorFromHue((float)i / parcels.size());
            //         render::draw_bold_polyline2(parcels[i].points, RL_BLACK, 0.f, pathWidth / 3.f, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
            //         render::fill_polygon2(Polyloop2(parcels[i].points), c, 0.f, 0.3f, {terrain.getWidth() * terrain.getCellSize(), 0});
            //         render::draw_bold_polyline2(parcelObbs[i].points, RL_RED, 0.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
            //     }
            // }

            // render::draw_bold_polyline2(testPolyA.points, RL_RED, 30.f, pathWidth, lineData.color.a);
            // for (const auto &p : weightOffsetPoly)
            // {
            //     render::draw_bold_polyline2(p.points, RL_RED, 30.f, pathWidth, lineData.color.a);
            // }
            // render::draw_bold_polyline2(averageOffsetPoly.points, RL_GREEN, 10.f, pathWidth, lineData.color.a);

            // for (const auto &poly : subedPolys)
            // {
            //     render::draw_bold_polyline2(poly.points, RL_DARKGREEN, 40.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
            // }

            // for (const auto &poly : interPolys)
            // {
            //     render::draw_bold_polyline2(poly.points, RL_DARKBLUE, 50.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
            // }

            // for (const auto &poly : cuttedPolys)
            // {
            //     render::draw_bold_polyline2(poly.points, RL_DARKPURPLE, 60.f, pathWidth, lineData.color.a, {terrain.getWidth() * terrain.getCellSize(), 0});
            // }

            // for (int i = 0; i < divPolys.size(); i++)
            // {
            //     Color c = renderUtil::ColorFromHue((float)i / divPolys.size());
            //     render::draw_bold_polyline2(divPolys[i].points, RL_DARKPURPLE, 0.f, pathWidth, lineData.color.a);
            //     render::fill_polygon2(divPolys[i].points, c, 0.f, 0.3f);
            // }
            //  for (const auto &poly : divPolys)
            // {
            //     render::draw_bold_polyline2(poly.points, RL_DARKPURPLE, 60.f, pathWidth, lineData.color.a /*,{terrain.getWidth() * terrain.getCellSize(), 0}*/);
            // }
            // render::draw_bold_polyline2(river.points, RL_RED, 0.f, pathWidth, lineData.color.a /*,{terrain.getWidth() * terrain.getCellSize(), 0}*/);
            // extrudeMesh.draw(color, colorAlpha,outline, wireframe, wireframeAlpha);
            // extrudeMesh_2.draw(GREEN, colorAlpha,outline, wireframe, wireframeAlpha);
            // extrudeMesh_3.draw(RED, colorAlpha,outline, wireframe, wireframeAlpha);
            // render::fill_polygon3(polyloop, RED, 0.5f);
            // render::stroke_bold_polygon3(polyloop, BLACK);
            for (int i = 0; i < tensorField.getAllTensors().size(); ++i)
            {
                const field::Tensor<float> &t = tensorField.getAllTensors()[i];
                if (i < (tensorField.getGridNX() + 1) * (tensorField.getGridNY() + 1))
                    for (int i = 0; i < 4; ++i)
                        render::draw_vector(t.pos, t.dirs[i], RL_BLACK, vecData.scale, vecData.startThickness, vecData.endThickness, render.vecData.vecZ, vecData.color.a);
                else
                    for (int i = 0; i < 4; ++i)
                        render::draw_vector(t.pos, t.dirs[i], RL_RED, vecData.scale, vecData.startThickness, vecData.endThickness, render.vecData.vecZ, vecData.color.a);
            }
            if (showAtrractor)
            {
                for (const auto &attr : attractors)
                    attr.draw(-76.37f);
            }

        },
        [&]() { // 二维屏幕空间绘图
            // render.draw_index_fonts(vertices, 16, BLUE);
            // if (showPlanarRoads)
            // {
            //     render.draw_index_fonts(nodePoints, render.fontData.size, render.fontData.color);
            // }

            terrain.drawContourPtIndices(layers, render);
            if (showText)
            {
                // for (const auto &parcel : parcels)
                // {
                //     render.draw_index_fonts(parcel.points, fontData.size, fontData.color, 0.f, {terrain.getWidth() * terrain.getCellSize(), 0});
                // }
                render.draw_index_fonts(volumeCenters, render.fontData.size, render.fontData.color);
            }

            rlImGuiBegin(); // 开始ImGui帧渲染（必须在2D阶段调用）
            // bool demoOpen = true;
            // ImGui::ShowDemoWindow(&demoOpen);

            // 2. 自定义GUI窗口（纯2D固定在屏幕上）
            bool customOpen = true;
            render.setCameraUI(customOpen);
            render.setDrawGeoDataUI(customOpen);
            if (ImGui::Begin("Render Settings", &customOpen))
            {
                // if (ImGui::CollapsingHeader("Camera Control", ImGuiTreeNodeFlags_DefaultOpen))
                // {
                //     ImGui::SliderFloat("Camera Move Speed", &render::RENDER_MOVE_SPEED, 0.f, 2.f, "%.2f");
                //     ImGui::SliderFloat("Camera Rotate Speed", &render::RENDER_ROTATE_SPEED, 0.f, 0.009f, "%.3f");
                //     ImGui::SliderFloat("Camera Zoom Speed", &render::RENDER_ZOOM_SPEED, 0.f, 10.f, "%.2f");
                // }

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

            if (ImGui::Begin("Refine Volume", &customOpen))
            {

                if (ImGui::TreeNodeEx("VolumeGen", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::InputInt("reBuildIndex", &reBuildIndex, 1, 400, 0);

                    if (ImGui::Button("Rebuild Volume"))
                    {

                        if (reBuildIndex >= 0 && reBuildIndex < volumes.size())
                        {

                            volumes[reBuildIndex].rebuild(terrain);
                        }
                    }

                    if (ImGui::Button("Export Obj"))
                    {
                        for (int i = 0; i < volumes.size(); i++)
                        {
                            const auto &volume = volumes[i];
                            volume.exportVolumePts2text(i, outputDir);
                            volume.exportYardPts2text(i, outputDir);
                        }
                        export3Dpts(projSCAroads,outputDir); 
                       
                    }

                    ImGui::TreePop();
                }
            }
            ImGui::End();

            if (ImGui::Begin("Terrain Info", &customOpen))
            {
                ImGui::Checkbox("Show Terrain", &showTerrain);
                ImGui::Text("Terrain size: %d x %d", ((int)terrain.getCellSize() * terrain.getWidth()), ((int)terrain.getCellSize() * terrain.getWidth()));

                // =====================================================
                // Generation Settings
                // =====================================================
                if (ImGui::CollapsingHeader("Generate Settings", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Indent();
                    needGenTerrain |= ImGui::SliderInt("Terrain Width(2^N)", &terrainPow, 5, 11);
                    needGenTerrain |= ImGui::SliderFloat("Frequency", &frequency, 0.f, 1.f, "%.2f");
                    needGenTerrain |= ImGui::SliderFloat("Amplitude", &amplitude, 0.f, 50.f, "%.2f");
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
                        // result = build_voronoi_from_pslg(Polyline2_t<float>(terrainBounding2, true), pathPolylines);
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
                    ImGui::RadioButton("ViewShed", &mode, 4);
                    ImGui::RadioButton("PointViewShed", &mode, 5);
                    ImGui::Indent();
                    ImGui::Checkbox("ShowViewPt", &terrain.showTestViewPt);
                    ImGui::Unindent();
                    ImGui::RadioButton("Flow", &mode, 6);
                    ImGui::RadioButton("Score", &mode, 7);
                    if (ImGui::Button("Apply River Color"))
                    {
                        terrain.applyPolyColor({river}, {179, 215, 216, 255});
                    }
                    if (ImGui::Button("Apply Yard Color"))
                    {
                        terrain.applyPolyColor(yardPolys, {175, 212, 120, 255});
                    }
                    if (ImGui::Button("BuildRealSite"))
                    {
                        terrain = Terrain(heightMap, 256, 256, 2.f);
                    }
                    ImGui::SliderFloat("OutlineThickness", &OutlineThickness, 0.f, 2.f, "%.2f");
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
                    ImGui::Checkbox("ShowProjectedRoads", &showProjectedRoads);
                    ImGui::Checkbox("ShowPlanarRoads", &showPlanarRoads);
                    ImGui::Checkbox("ShowPlanarSites", &showPlanarSites);
                    ImGui::Checkbox("ShowVolumes", &showVolumes);
                    ImGui::Checkbox("ShowYards", &showYardVolumes);
                    ImGui::Checkbox("ShowAttractors", &showAtrractor);
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
                    offsetInside |= ImGui::SliderFloat("OffsetDistInside", &offsetDist2, -10.f, 10.f, "%.1f");
                    offsetInside |= ImGui::SliderFloat("BuildingDepth", &buildingDepth, -10.f, 10.f, "%.1f");
                    ImGui::Unindent();
                }
            }
            ImGui::End();

            rlImGuiEnd();
        }});

    return 0;
}