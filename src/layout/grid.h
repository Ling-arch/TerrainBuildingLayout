#pragma once

#include "geo.h"


namespace grid
{

    enum class EdgeDirection
    {
        Up,
        Right,
        Down,
        Left
    };

    template <typename Scalar>
    struct InverseTran
    {
        Eigen::Vector2<Scalar> center = Eigen::Vector2<Scalar>::Zero();
        Eigen::Matrix<Scalar, 2, 2> Rinv = Eigen::Matrix<Scalar, 2, 2>::Identity();

        InverseTran() = default;

        // 确保传入的 R 是 2x2 矩阵
        InverseTran(const Eigen::Vector2<Scalar> &cen, const Eigen::Matrix<Scalar, 2, 2> &R)
            : center(cen)
        {
            // 确保R是一个2x2矩阵
            assert(R.rows() == 2 && R.cols() == 2); // 这行是可选的，帮助检查传入矩阵的尺寸

            // 如果传入的 R 不是单位矩阵，则进行逆运算
            Rinv = R.inverse();
        }
    };

    struct BoundaryEdge
    {
        int cellIdx = -1;  // 边所在的cell索引
        int dir = -1;      // 0= up, 1=right, 2=down, 3=left
        int neighbor = -1; // -1表示没有邻居
        BoundaryEdge() = default;
        BoundaryEdge(int _cellIdx, int _dir) : cellIdx(_cellIdx), dir(_dir) {}
        BoundaryEdge(int _cellIdx, int _dir, int _neighbor) : cellIdx(_cellIdx), dir(_dir), neighbor(_neighbor) {}
    };

    struct ContourSegment
    {
        std::vector<BoundaryEdge> segments; // 对应边
        int dir;                            // 0= up, 1=right, 2=down, 3=left
        bool startConvex = false;           // 起点是否是凸点
        bool endConvex = false;             // 终点是否是凸点
    };

    struct GridCell
    {
        Eigen::Vector2f center; // 实际坐标
        Eigen::Vector2i coord;  // 二维坐标索引
        int id = -1;
        // 邻接（4方向）的索引,-1表示没有邻居
        int neighbors[4] = {-1, -1, -1, -1}; // Up Right Down Left
        geo::Polyline2_t<float> cellPoly;
        GridCell() = default;
        GridCell(const Eigen::Vector2f &cen, float inSize) : center(cen)
        {
            std::vector<Eigen::Vector2f> corners;
            corners.resize(4);
            float half = inSize / 2.0f;

            // 顺序：左上 → 右上 → 右下 → 左下
            corners[0] = center + Eigen::Vector2f(-half, half);  // 左上
            corners[1] = center + Eigen::Vector2f(half, half);   // 右上
            corners[2] = center + Eigen::Vector2f(half, -half);  // 右下
            corners[3] = center + Eigen::Vector2f(-half, -half); // 左下
            cellPoly = corners;
        }

        inline std::pair<Eigen::Vector2f, Eigen::Vector2f> getEdge(int dir) const
        {
            const auto &p = cellPoly.points;

            if (dir == 0)
                return {p[0], p[1]}; // Up
            if (dir == 1)
                return {p[1], p[2]}; // Right
            if (dir == 2)
                return {p[2], p[3]}; // Down
            return {p[3], p[0]};     // Left
        }
    };

    // 局部的一个区域内的cells
    struct CellGroup
    {
        int id;

        std::vector<int> cellIndices;             // ⭐ 核心：只存 index
        const std::vector<GridCell> *globalCells; // 指向全局
        std::vector<ContourSegment> contourSegments;
        geo::Polyline2_t<float> contourPoly;
        std::vector<geo::Polyline2_t<float>> contourPolys;
        CellGroup() = default;
        CellGroup(const std::vector<int> &indices,
                  const std::vector<GridCell> *global,
                  int inId)
            : cellIndices(indices), globalCells(global), id(inId)
        {
            if (globalCells)
            {
                for (int idx : cellIndices)
                {
                    // 防越界
                    if (idx < 0 || idx >= (int)globalCells->size())
                        continue;

                    // 修改 global cell 的归属 id
                    const_cast<GridCell &>((*globalCells)[idx]).id = id;
                }
            }

            buildContourSegments();

            buildContour();
        }

        void buildContourSegments();
        void buildContour();
        void buildMultipleContours();
        std::vector<std::vector<int>> findRectGroups() const;
    };

    // 管理多个cell group的边界关系，构建轮廓网格
    struct CellRegion
    {
        const std::vector<GridCell> *globalCells;
        std::vector<CellGroup> groups;
        std::vector<geo::PolygonMesh> contourMeshes;
        std::vector<std::vector<int>> rebuildIndices;

        CellRegion(int dim,
                   const std::vector<GridCell> *global,
                   const std::vector<std::vector<int>> &groupIndices,
                   const std::vector<float> &baseHeights,
                   const std::vector<int> &floors,
                   const std::vector<int> &isAffect)
            : globalCells(global)
        {
            int id = 0;
            groups.clear();
            for (const auto &indices : groupIndices)
            {
                groups.emplace_back(indices, globalCells, id++);
            }
            
            if (dim == 3)
            {
                buildContourMeshes(baseHeights, floors);
            }
            // swapEdgeCells();
            mergeSingleCell();
        }

        CellRegion() = default;
        void swapEdgeCells();
        void mergeSingleCell();
        void pushAdditionalCells();
        void buildContourMeshes(const std::vector<float> &baseHeights, const std::vector<int> &floors);
    };

    struct FloorSystem
    {
        const std::vector<GridCell> *globalCells = nullptr;
        struct Layer
        {
            float height = 0.0f;
            std::vector<int> cellIndices;
        };
        std::vector<geo::PolygonMesh> floorMeshes;
        std::vector<geo::PolygonMesh> yardMeshes;
        std::vector<geo::Polyline2_t<float>> volumePolys;
        std::vector<geo::Polyline2_t<float>> yardPolys;
        geo::MeshData meshData; // CPU mesh
        Model model;            //
        FloorSystem() = default;
        FloorSystem(float targetFar,const std::vector<GridCell> *cells, const std::vector<std::vector<int>> &groupIndices,
                    const std::vector<float> &baseHeights,
                    const std::vector<int> &floors,
                    const std::vector<int> &isAffect,
                    const geo::MeshData &originalMesh)
            : globalCells(cells)
        {
            build(targetFar,groupIndices, baseHeights, floors, isAffect, originalMesh);
        }

        FloorSystem(const InverseTran<float> &inverseTran, float targetFar, const std::vector<GridCell> *cells, const std::vector<std::vector<int>> &groupIndices,
                    const std::vector<float> &baseHeights,
                    const std::vector<int> &floors,
                    const std::vector<int> &isAffect,
                    const geo::MeshData &originalMesh)
            : globalCells(cells)
        {
            build(targetFar, groupIndices, baseHeights, floors, isAffect, originalMesh, inverseTran);
        }

        void build(
            float targetFar,
            const std::vector<std::vector<int>> &groupIndices,
            const std::vector<float> &baseHeights,
            const std::vector<int> &floors,
            const std::vector<int> &isAffect,
            const geo::MeshData &originalMesh,const InverseTran<float> &inverseTran= InverseTran<float>());

        void buildChangedTerrainMesh(
            float targetFar,
            const geo::MeshData &originalMesh,
            const std::map<float, Layer> &layerCells,
            const std::vector<std::vector<int>> &yardCells,
            const std::vector<float> &yardHeights,
            const std::unordered_set<int> &floatingCells, const InverseTran<float> &inverseTran = InverseTran<float>());

        // void mergeSingleCell();
        void drawTerrain(Color color, float colorAlpha, bool wireframe, float wireframeAlpha, Eigen::Vector3f position = {0.f, 0.f, 0.f}) const;
     
    };

    // 全局的Cell数组
    // 管理全局的cell的邻接关系
    struct CellGenerator
    {
        std::vector<GridCell> cells;
        float cellSize;
        std::unordered_map<int64_t, int> coordMap; // 二维坐标到cell索引的映射

        CellGenerator() = default;
        CellGenerator(const geo::Polyline2_t<float> &site, float cellSize_)
            : cellSize(cellSize_)
        {
            generateCells(site);
            buildCellNeighbors();
        }

        void generateCells(const geo::Polyline2_t<float> &site);
        void buildCellNeighbors();
    };

    inline int64_t encode(int x, int y)
    {
        return (int64_t(x) << 32) | (uint32_t)y;
    }
}