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

            // std::cout << "[INFO] building height adjusted to avoid terrain\n";
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
                  << " POINTS NUM " << buildingPlanPoints.size()
                  << " extrude: " << randomHeight << "\n";

        buildingMesh = PolygonMesh(buildingPlanPoints, randomHeight);
    }

    void Volume::build(const Polyline2_t &site, const terrain::Terrain &terrain)
    {
        // 计算 OBB（有界包围盒）和旋转矩阵
        // geo::OBB2<float> obb(site.points);
        // std::cout << "OBB Axis 0: (" << obb.axis0.x() << ", " << obb.axis0.y() << ")\n";

        // // 计算旋转矩阵
        // Eigen::Matrix<float, 2, 2> R = geo::rotationToXAxis(obb.axis0);
        // Eigen::Matrix<float, 2, 2> Rinv = R.transpose();

        // std::cout << "Rotation Matrix R:\n"
        //           << R << "\n";
        // std::cout << "Inverse Rotation Matrix Rinv:\n"
        //           << Rinv << "\n";

        // // 旋转多边形
        // rotPoly = geo::rotatePoly(site, R);
        // std::cout << "Rotated Polygon (rotPoly):\n";
        // for (const auto &p : rotPoly.points)
        // {
        //     std::cout << "(" << p.x() << ", " << p.y() << ")\n";
        // }

        // // 计算最大矩形
        //  temp = geo::getMaxRectInPolyWithRatio(rotPoly, 0.5, 1.8);
        // std::cout << "Max Rect Polygon (temp):\n";
        // for (const auto &p : temp.points)
        // {
        //     std::cout << "(" << p.x() << ", " << p.y() << ")\n";
        // }

        // // 将最大矩形的点旋转回原始坐标系
        // std::vector<Eigen::Vector2<float>> originalPts;
        // for (const auto &p : temp.points)
        // {
        //     originalPts.emplace_back(Rinv * p);
        //     std::cout << "Original Point (after reverse rotation): ("
        //               << originalPts.back().x() << ", "
        //               << originalPts.back().y() << ")\n";
        // }

        // // 更新 rectSite
        // rectSite = Polyline2_t(originalPts, true);

        /* BuildingLayout */ layout = BuildingLayout(site, terrain);
        std::cout << "Base Area is :" << util::Math2<float>::polygon_area(layout.rotedSite.points) << "\n";
        field::Polyline2_t<float> boundOffset = geo::offsetPolygon(layout.rotedSite, -1.5f)[0]; // 以旋转后的rect生成
        std::vector<Eigen::Vector2f> yardSeeds = geo::samplePointsOnPolygonWithSpacing(boundOffset, 2, (unsigned)time(nullptr));
        util::Math2<float>::PoissonResult totalSeedResult = util::Math2<float>::gen_poisson_sites_in_poly_with_seeds(layout.rotedSite.points, yardSeeds, layout.divGap, 6, 30, (unsigned)time(nullptr));
        torch::Tensor grid_xy = diffVoronoi::vec2_to_tensor(layout.rotedCenters); // 旋转后的格子中心点
        torch::Tensor site_xy = diffVoronoi::vec2_to_tensor(totalSeedResult.samples);
        std::cout << "sample point num is " << totalSeedResult.samples.size() << "\n";
        torch::Tensor terrain_h = torch::from_blob(layout.heightMap.data(), {static_cast<int64_t>(layout.heightMap.size())}).clone();
        std::cout << "girds num is : " << layout.heightMap.size() << "\n"; // 旋转后的格子中心点的高度值
        layout::SoftRVDModel softModel = layout::SoftRVDModel(grid_xy, terrain_h, site_xy, 1.2f, {0, 1}, {1, 1, 0, 1, 1}, 20.f, 10.f);
        auto start = std::chrono::high_resolution_clock::now();
        softModel.optimize();
        // 时间记录：结束时间
        auto end = std::chrono::high_resolution_clock::now();

        // 计算时间差（单位：毫秒）
        std::chrono::duration<double> duration = end - start;
        std::cout << "Optimization took: " << duration.count() << " seconds." << std::endl;
        grid::CellGenerator cellGen(layout.rotedSite, 1.f);
        std::pair<grid::CellRegion, grid::FloorSystem> floorCellVolumeLayers = softModel.buildCellRegion(cellGen, layout.meshData, layout.inverseTran);
        const auto &floorManager = floorCellVolumeLayers.second;
        volumeMeshes.clear();
        yardMeshes.clear();

        yardBounds.clear();
        volumePolys.clear();
        yardPolys.clear();
        yardBounds.insert(yardBounds.end(), floorManager.yardPolys.begin(), floorManager.yardPolys.end());

        maxHeight = -std::numeric_limits<float>::infinity();
        for (const auto &v : floorManager.floorMeshes)
        {
            volumeMeshes.push_back(v);
            const std::vector<Eigen::Vector3f> &points = v.getPoints();
            volumePolys.push_back(points);
            if (!points.empty())
            {
                float z = points[0].z();
                maxHeight = std::max(maxHeight, z);
            }
        }
        maxHeight += 4.0f;

        for (const auto &y : floorManager.yardMeshes)
        {
            yardMeshes.push_back(y);

            yardPolys.push_back(y.getPoints());
        }
    }

    void Volume::rebuild(const terrain::Terrain &terrain)
    {
        /* BuildingLayout */ layout = BuildingLayout(site, terrain);
        std::cout << "Base Area is :" << util::Math2<float>::polygon_area(layout.rotedSite.points) << "\n";
        field::Polyline2_t<float> boundOffset = geo::offsetPolygon(layout.rotedSite, -1.5f)[0]; // 以旋转后的rect生成
        std::vector<Eigen::Vector2f> yardSeeds = geo::samplePointsOnPolygonWithSpacing(boundOffset, 2, (unsigned)time(nullptr));
        util::Math2<float>::PoissonResult totalSeedResult = util::Math2<float>::gen_poisson_sites_in_poly_with_seeds(layout.rotedSite.points, yardSeeds, layout.divGap, 6, 30, (unsigned)time(nullptr));
        torch::Tensor grid_xy = diffVoronoi::vec2_to_tensor(layout.rotedCenters); // 旋转后的格子中心点
        torch::Tensor site_xy = diffVoronoi::vec2_to_tensor(totalSeedResult.samples);
        std::cout << "sample point num is " << totalSeedResult.samples.size() << "\n";
        torch::Tensor terrain_h = torch::from_blob(layout.heightMap.data(), {static_cast<int64_t>(layout.heightMap.size())}).clone();
        std::cout << "girds num is : " << layout.heightMap.size() << "\n"; // 旋转后的格子中心点的高度值
        layout::SoftRVDModel softModel = layout::SoftRVDModel(grid_xy, terrain_h, site_xy, 1.2f, {0, 1}, {1, 1, 0, 1, 1}, 20.f, 10.f);
        auto start = std::chrono::high_resolution_clock::now();
        softModel.optimize();
        // 时间记录：结束时间
        auto end = std::chrono::high_resolution_clock::now();

        // 计算时间差（单位：毫秒）
        std::chrono::duration<double> duration = end - start;
        std::cout << "Optimization took: " << duration.count() << " seconds." << std::endl;
        grid::CellGenerator cellGen(layout.rotedSite, 1.f);
        std::pair<grid::CellRegion, grid::FloorSystem> floorCellVolumeLayers = softModel.buildCellRegion(cellGen, layout.meshData, layout.inverseTran);
        const auto &floorManager = floorCellVolumeLayers.second;
        volumeMeshes.clear();
        yardMeshes.clear();

        yardBounds.clear();
        volumePolys.clear();
        yardPolys.clear();
        yardBounds.insert(yardBounds.end(), floorManager.yardPolys.begin(), floorManager.yardPolys.end());

        maxHeight = -std::numeric_limits<float>::infinity();
        for (const auto &v : floorManager.floorMeshes)
        {
            volumeMeshes.push_back(v);
            const std::vector<Eigen::Vector3f> &points = v.getPoints();
            volumePolys.push_back(points);
            if (!points.empty())
            {
                float z = points[0].z();
                maxHeight = std::max(maxHeight, z);
            }
        }
        maxHeight += 4.0f;
        for (const auto &v : floorManager.floorMeshes)
        {
            volumeMeshes.push_back(v);
            volumePolys.push_back(v.getPoints());
        }

        for (const auto &y : floorManager.yardMeshes)
        {
            yardMeshes.push_back(y);

            yardPolys.push_back(y.getPoints());
        }
    }

    void Volume::exportVolumePts2text(int index, const std::string &outputDir)const
    {
        for (size_t i = 0; i < volumePolys.size(); ++i)
        {
            std::string filename = outputDir + "/Volume_" + std::to_string(index) + "_" + std::to_string(i) + ".txt";
            std::ofstream outFile(filename); // 打开文件进行写入

            if (!outFile.is_open())
            {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            // 遍历当前 Polyline2_t 的所有点并写入文件
            const Poly3 &poly = volumePolys[i];
            for (const auto &point : poly)
            {

                outFile << point.x() << "," << point.y() << "," << point.z() << std::endl;
            }

            outFile.close(); // 关闭文件
            // std::cout << "Exported points to: " << filename << std::endl;
        }
    }
    void Volume::exportYardPts2text(int index, const std::string &outputDir)const
    {
        for (size_t i = 0; i < yardPolys.size(); ++i)
        {
            std::string filename = outputDir + "/Yard_" + std::to_string(index) + "_" + std::to_string(i) + ".txt";
            std::ofstream outFile(filename); // 打开文件进行写入

            if (!outFile.is_open())
            {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            const Poly3 &poly = yardPolys[i];
            for (const auto &point : poly)
            {

                outFile << point.x() << "," << point.y() << "," << point.z() << std::endl;
            }

            outFile.close(); // 关闭文件
            // std::cout << "Exported points to: " << filename << std::endl;
        }
    }

    void ExportMeshToObj(const Mesh &mesh, const char *filename)
    {
        std::ofstream file(filename);

        if (!file.is_open())
        {
            std::cerr << "Error: Unable to open file for writing!" << std::endl;
            return;
        }

        // 写入 OBJ 文件头
        file << "# Exported by Raylib\n\n";

        // 写入顶点（v）
        for (int i = 0; i < mesh.vertexCount; i++)
        {
            // 解引用指针来访问数据
            float *vertex = &mesh.vertices[i * 3]; // 每个顶点有 3 个浮动值 (X, Y, Z)
            file << "v " << vertex[0] << " " << vertex[1] << " " << vertex[2] << "\n";
        }

        // 写入法线（vn）
        if (mesh.normals != nullptr)
        {
            for (int i = 0; i < mesh.vertexCount; i++)
            {
                // 解引用指针来访问法线数据
                float *normal = &mesh.normals[i * 3]; // 每个法线有 3 个浮动值 (X, Y, Z)
                file << "vn " << normal[0] << " " << normal[1] << " " << normal[2] << "\n";
            }
        }

        // 写入纹理坐标（vt）
        if (mesh.texcoords != nullptr)
        {
            for (int i = 0; i < mesh.vertexCount; i++)
            {
                // 解引用指针来访问纹理坐标数据
                float *texcoord = &mesh.texcoords[i * 2]; // 每个纹理坐标有 2 个浮动值 (U, V)
                file << "vt " << texcoord[0] << " " << texcoord[1] << "\n";
            }
        }

        // 写入面（f）
        for (int i = 0; i < mesh.triangleCount; i++)
        {
            int index0 = mesh.indices[i * 3 + 0] + 1; // OBJ 文件索引从 1 开始
            int index1 = mesh.indices[i * 3 + 1] + 1;
            int index2 = mesh.indices[i * 3 + 2] + 1;

            // 如果有纹理坐标和法线
            if (mesh.texcoords != nullptr && mesh.normals != nullptr)
            {
                file << "f " << index0 << "/" << index0 << "/" << index0 << " "
                     << index1 << "/" << index1 << "/" << index1 << " "
                     << index2 << "/" << index2 << "/" << index2 << "\n";
            }
            // 如果没有纹理坐标，但有法线
            else if (mesh.normals != nullptr)
            {
                file << "f " << index0 << "//" << index0 << " "
                     << index1 << "//" << index1 << " "
                     << index2 << "//" << index2 << "\n";
            }
            // 如果没有法线，但有纹理坐标
            else if (mesh.texcoords != nullptr)
            {
                file << "f " << index0 << "/" << index0 << " "
                     << index1 << "/" << index1 << " "
                     << index2 << "/" << index2 << "\n";
            }
            // 如果没有纹理坐标和法线
            else
            {
                file << "f " << index0 << " " << index1 << " " << index2 << "\n";
            }
        }

        file.close();
        std::cout << "Mesh exported to " << filename << std::endl;
    }

    void ExportModelToObj(const Model &model, const char *filenamePrefix)
    {
        for (int i = 0; i < model.meshCount; i++)
        {
            Mesh mesh = model.meshes[i];
            std::string filename = std::string(filenamePrefix) + "_mesh" + std::to_string(i) + ".obj";
            ExportMeshToObj(mesh, filename.c_str());
        }
    }
}