#include "grid.h"

namespace grid
{

    void CellGroup::buildContourSegments()
    {
        contourSegments.clear();

        if (!globalCells)
            return;

        // 1️ group set（用于判断是否在 group 内）
        std::unordered_set<int> groupSet(cellIndices.begin(), cellIndices.end());

        // 2️ 收集 boundary edges
        std::vector<BoundaryEdge> edges;

        for (int idx : cellIndices)
        {
            const auto &c = globalCells->at(idx);

            for (int d = 0; d < 4; ++d)
            {
                int nb = c.neighbors[d];

                if (nb == -1 || groupSet.count(nb) == 0)
                {
                    edges.emplace_back(idx, d, nb);
                }
            }
        }

        if (edges.empty())
            return;

        // 3️ 建立 (coord, dir) → edge index 的映射
        std::unordered_map<int64_t, int> edgeMap;

        auto encodeEdge = [](int x, int y, int dir)
        {
            return (int64_t(x) << 32) | (uint32_t(y << 2 | dir));
        };

        for (int i = 0; i < edges.size(); ++i)
        {
            const auto &e = edges[i];
            const auto &c = globalCells->at(e.cellIdx);

            edgeMap[encodeEdge(c.coord.x(), c.coord.y(), e.dir)] = i;
        }

        // 4️ 找起点（左下角优先）
        int start = 0;
        {
            int minX = INT_MAX, minY = INT_MAX;

            for (int i = 0; i < edges.size(); ++i)
            {
                const auto &c = globalCells->at(edges[i].cellIdx);

                if (c.coord.y() < minY ||
                    (c.coord.y() == minY && c.coord.x() < minX))
                {
                    minX = c.coord.x();
                    minY = c.coord.y();
                    start = i;
                }
            }
        }

        // 5️ 按拓扑走 loop（O(N)）
        std::vector<BoundaryEdge> ordered;
        std::vector<bool> used(edges.size(), false);

        int cur = start;

        while (true)
        {
            if (used[cur])
                break;

            used[cur] = true;
            const auto &e = edges[cur];
            ordered.push_back(e);

            const auto &c = globalCells->at(e.cellIdx);
            int x = c.coord.x();
            int y = c.coord.y();
            int d = e.dir;

            //  找下一条（基于 coord）
            int nx = x, ny = y, nd = d;

            if (d == 0)
            {
                nx = x + 1;
                ny = y;
            } // Up
            else if (d == 1)
            {
                nx = x;
                ny = y - 1;
            } // Right
            else if (d == 2)
            {
                nx = x - 1;
                ny = y;
            } // Down
            else if (d == 3)
            {
                nx = x;
                ny = y + 1;
            } // Left

            auto it = edgeMap.find(encodeEdge(nx, ny, nd));

            if (it == edgeMap.end())
                break;

            cur = it->second;
        }

        // 6️ 按方向分段（你要的逻辑 ✔）
        ContourSegment current;
        current.dir = ordered[0].dir;

        for (const auto &e : ordered)
        {
            if (e.dir == current.dir)
            {
                current.segments.push_back(e);
            }
            else
            {
                contourSegments.push_back(current);

                current = ContourSegment();
                current.dir = e.dir;
                current.segments.push_back(e);
            }
        }

        if (!current.segments.empty())
            contourSegments.push_back(current);
    }

    void CellGroup::buildContour()
    {
        contourPoly.points.clear();

        if (contourSegments.empty() || globalCells == nullptr)
            return;

        std::vector<Eigen::Vector2f> pts;
        pts.reserve(contourSegments.size() + 1);

        for (int i = 0; i < contourSegments.size(); ++i)
        {
            const auto &seg = contourSegments[i];

            if (seg.segments.empty())
                continue;

            const BoundaryEdge &first = seg.segments.front();
            const BoundaryEdge &last = seg.segments.back();

            const GridCell &c0 = globalCells->at(first.cellIdx);
            const GridCell &c1 = globalCells->at(last.cellIdx);

            // 当前 segment 的起点 / 终点
            auto [p0_start, p0_end] = c0.getEdge(first.dir);
            auto [p1_start, p1_end] = c1.getEdge(last.dir);

            // segment 起点
            Eigen::Vector2f start = p0_start;

            // segment 终点
            Eigen::Vector2f end = p1_end;

            // 第一段初始化
            if (pts.empty())
            {
                pts.push_back(start);
            }
            else
            {
                // 防止重复点（非常重要）
                if ((pts.back() - start).squaredNorm() > 1e-6)
                    pts.push_back(start);
            }

            pts.push_back(end);
        }

        // 闭合处理（确保首尾一致）
        if (!pts.empty())
        {
            if ((pts.front() - pts.back()).squaredNorm() > 1e-6)
                pts.push_back(pts.front());
        }

        contourPoly = geo::Polyline2_t<float>(pts, true);
    }

    void CellGroup::debugPrintSegments() const
    {
        if (!globalCells)
        {
            std::cout << "[Debug] globalCells is null\n";
            return;
        }

        std::cout << "================ Contour Segments ================\n";

        for (int i = 0; i < contourSegments.size(); ++i)
        {
            const auto &seg = contourSegments[i];

            std::cout << "Segment " << i << " dir = " << seg.dir
                      << " size = " << seg.segments.size() << "\n";

            if (seg.segments.empty())
                continue;

            const auto &first = seg.segments.front();
            const auto &last = seg.segments.back();

            const GridCell &c0 = globalCells->at(first.cellIdx);
            const GridCell &c1 = globalCells->at(last.cellIdx);

            auto [s0, s1] = c0.getEdge(first.dir);
            auto [e0, e1] = c1.getEdge(last.dir);

            std::cout << "  StartCellIdx: " << first.cellIdx
                      << " coord=(" << c0.coord.x() << "," << c0.coord.y() << ")"
                      << " startPt=(" << s0.x() << "," << s0.y() << ")\n";

            std::cout << "  EndCellIdx:   " << last.cellIdx
                      << " coord=(" << c1.coord.x() << "," << c1.coord.y() << ")"
                      << " endPt=(" << e1.x() << "," << e1.y() << ")\n";
        }

        std::cout << "===================================================\n";
    }

    void CellGroup::debugPrintContour() const
    {
        std::cout << "================ Contour Polyline ================\n";

        const auto &pts = contourPoly.points;

        for (int i = 0; i < pts.size(); ++i)
        {
            std::cout << "P[" << i << "] = ("
                      << pts[i].x() << ", "
                      << pts[i].y() << ")\n";
        }

        std::cout << "Point count = " << pts.size() << "\n";

        if (!pts.empty())
        {
            float dx = pts.front().x() - pts.back().x();
            float dy = pts.front().y() - pts.back().y();

            std::cout << "Closed check: "
                      << (std::sqrt(dx * dx + dy * dy) < 1e-6 ? "YES" : "NO")
                      << "\n";
        }

        std::cout << "==================================================\n";
    }

    void CellGenerator::generateCells(const geo::Polyline2_t<float> &site) 
    {

        cells.clear();
        const Eigen::AlignedBox2f &bound = site.getAABB2();
        float minX = bound.min().x();
        float maxX = bound.max().x();
        float minY = bound.min().y();
        float maxY = bound.max().y();

        int nx = static_cast<int>((maxX - minX) / cellSize) + 1;
        int ny = static_cast<int>((maxY - minY) / cellSize) + 1;

        for (int iy = 0; iy < ny; ++iy)
        {
            for (int ix = 0; ix < nx; ++ix)
            {
                Eigen::Vector2f center(minX + (ix + 0.5f) * cellSize,minY + (iy + 0.5f) * cellSize);

                if (!util::Math2<float>::point_in_poly(site.points, center))
                    continue;

                GridCell cell(center, cellSize);
                cell.coord = {ix, iy};
                cells.push_back(cell);
            }
        }
    }

    void CellGenerator::buildCellNeighbors()
    {
        coordMap.clear();

        // 1️ 建表
        for (int i = 0; i < cells.size(); ++i)
        {
            const auto &c = cells[i];
            coordMap[encode(c.coord.x(), c.coord.y())] = i;
        }

        // 2️ 查邻居（O(1)）
        //上（0，1）右（1，0）下（0，-1）左（-1，0）
        const int dx[4] = {0, 1, 0, -1};
        const int dy[4] = {1, 0, -1, 0};

        for (int i = 0; i < cells.size(); ++i)
        {
            auto &cell = cells[i];

            for (int d = 0; d < 4; ++d)
            {
                int nx = cell.coord.x() + dx[d];
                int ny = cell.coord.y() + dy[d];

                auto it = coordMap.find(encode(nx, ny));
                if (it != coordMap.end())
                    cell.neighbors[d] = it->second;
            }
        }
    }
}