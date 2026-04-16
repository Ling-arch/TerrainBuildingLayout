#include "building.h"

namespace building
{

    void Building::buildBuildingMesh()
    {
        // =========================
        // 0. 基本保护
        // =========================
        if (layout.heightMap.empty())
        {
            std::cout << "[BUILDING ERROR] heightMap empty\n";
            return;
        }

        if (layout.oriRect.points.size() < 3)
        {
            std::cout << "[BUILDING ERROR] oriRect invalid\n";
            return;
        }

        float rectArea = util::Math2<float>::polygon_area(layout.oriRect.points);
        if (rectArea < 30.f || rectArea > 175.f)
            return;

        // =========================
        // 1. 地形高度范围
        // =========================
        const std::vector<float> &heightMap = layout.heightMap;

        float minHeight = *std::min_element(heightMap.begin(), heightMap.end());
        float maxHeight = *std::max_element(heightMap.begin(), heightMap.end());

        // =========================
        // 2. 随机建筑高度 [6, 10]
        // =========================
        uint32_t seed = std::hash<float>{}(layout.oriRect.points[0].x()) ^
                        std::hash<float>{}(layout.oriRect.points[0].y());

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(6.0f, 10.0f);

        float randomHeight = dist(rng);

        // =========================
        // 3. 计算最终建筑高度
        // =========================
        float buildingHeight = minHeight + randomHeight;

        if (buildingHeight < maxHeight)
        {
            // 如果建筑被地形遮挡，强制抬高
            randomHeight = (maxHeight - minHeight) + 3.0f;
            buildingHeight = minHeight + randomHeight;

            std::cout << "[INFO] building height adjusted to avoid terrain\n";
        }

        // =========================
        // 4. 构建底面点
        // =========================
        const Polyline2_t &rect = layout.oriRect;

        std::vector<Eigen::Vector3f> buildingPlanPoints;
        buildingPlanPoints.reserve(rect.points.size());

        for (const auto &p : rect.points)
        {
            buildingPlanPoints.emplace_back(p.x(), p.y(), minHeight);
        }

        // =========================
        // 5. 生成 mesh
        // =========================
        std::cout << "[BUILDING] minH: " << minHeight
                  << " maxH: " << maxHeight
                  << "POINTS NUM" << buildingPlanPoints.size()
                  << " extrude: " << randomHeight << "\n";

        buildingMesh = PolygonMesh(buildingPlanPoints, randomHeight);
    }
}