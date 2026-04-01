#pragma once

#include "geo.h"

namespace building
{
    enum class EdgeDirection{
        Up,
        Right,
        Down,
        Left
    };

    struct GridCell
    {
        Eigen::Vector2f center;
        float size;
        std::vector<Eigen::Vector2f> corners;
        int id;
        geo::Polyline2_t<float> cellPoly;
        GridCell() = default;
        GridCell(const Eigen::Vector2f &cen, float inSize) : center(cen), size(inSize)
        {
            corners.resize(4);
            float half = size / 2.0f;

            // 顺序：左上 → 右上 → 右下 → 左下
            corners[0] = center + Eigen::Vector2f(-half, half); // 左上
            corners[1] = center + Eigen::Vector2f(half, half);  // 右上
            corners[2] = center + Eigen::Vector2f(half, -half);   // 右下
            corners[3] = center + Eigen::Vector2f(-half, -half);  // 左下
            cellPoly = corners;
        }
    };

    struct Room2D
    {
        std::vector<GridCell> cells;
        int id;
        Room2D() = default;
        Room2D(const std::vector<GridCell> &inCells, int inId) : cells(inCells), id(inId)
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
        Building() = default;
        Building(const std::vector<Room3D> &inRooms) : rooms(inRooms)
        {
        }

        void arrangeRoomsToPlan();
    };
}