#pragma once

#include "geo.h"
#include "grid.h"
#include "layout.h"

namespace building
{

    struct Room2D
    {
        std::vector<grid::GridCell> cells;
        int id;
        Room2D() = default;
        Room2D(const std::vector<grid::GridCell> &inCells, int inId) : cells(inCells), id(inId)
        {
            for (auto &cell : cells)
                cell.id = id;
        }
    };

    struct Plan
    {
        std::vector<Room2D> rooms;
        int floor;
        std::vector<std::vector<int>> roomConnections;
        Plan() = default;
        Plan(const std::vector<Room2D> &inRooms, int inFloor) : rooms(inRooms), floor(inFloor)
        {
        }
        void calConnection();
    };

    struct Room3D
    {
        float area;
        float height;
        int id;
        Room3D() = default;
        Room3D(float inArea, float inHeight) : area(inArea), height(inHeight)
        {
        }
    };

    class Building
    {
    public:
        std::vector<Plan> plans;
        std::vector<Room3D> rooms;
        using PolygonMesh = geo::PolygonMesh;
        using Polyline2_t = geo::Polyline2_t<float>;
        Polyline2_t site;                      // 建筑占地轮廓
        std::vector<PolygonMesh> roomMeshes;   // 每个房间的可视化网格
        PolygonMesh buildingMesh;              // 建筑体量的简化网格
        std::vector<PolygonMesh> volumeMeshes; // 根据地形高差划分的体量网格
        using BuildingLayout = layout::BuildingLayout<float>;
        BuildingLayout layout; // 布局的地形信息

        Building() = default;
        Building(const Polyline2_t &site_, const terrain::Terrain &terrain) : site(site_)
        {
            layout = BuildingLayout(site, terrain);
            buildBuildingMesh();
        }
        Building(const std::vector<Room3D> &inRooms) : rooms(inRooms)
        {
        }

    public:
        void buildBuildingMesh();
        void arrangeRoomsToPlan();
    };

    class Volume
    {

    public:
        using PolygonMesh = geo::PolygonMesh;
        using Polyline2_t = geo::Polyline2_t<float>;
        using BuildingLayout = layout::BuildingLayout<float>;
        Polyline2_t rectSite;
        std::vector<PolygonMesh> volumeMeshes;
        std::vector<PolygonMesh> yardMeshes;
        std::vector<Polyline2_t> yardBounds;
        using Poly3 = typename std::vector<Eigen::Vector3f>;
        std::vector<Poly3> volumePolys;
        std::vector<Poly3> yardPolys;
        BuildingLayout layout;
        Polyline2_t site;
        
        float maxHeight;
        // layout::SoftRVDShowData showData;
        // layout::SoftRVDModel softModel;
        Volume() = default;
        Volume(const Polyline2_t &site_, const terrain::Terrain &terrain)
        :site(site_)
        {
            build(site_, terrain);
        };

        void build(const Polyline2_t &site, const terrain::Terrain &terrain);
        void rebuild(const terrain::Terrain &terrain);
        void exportVolumePts2text(int index, const std::string &outputDir)const;
        void exportYardPts2text(int index, const std::string &outputDir)const;
    };

    void ExportMeshToObj(const Mesh &mesh, const char *filename);
    void ExportModelToObj(const Model &model, const char *filename);
}