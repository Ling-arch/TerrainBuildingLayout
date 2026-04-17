#include "grid.h"

namespace grid
{

    void CellGroup::buildContourSegments()
    {
        contourSegments.clear();

        if (!globalCells)
            return;

        std::unordered_set<int> groupSet(cellIndices.begin(), cellIndices.end());

        std::vector<BoundaryEdge> edges;

        // =====================================================
        // 1. collect boundary edges
        // =====================================================
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

        // std::cout << "========== Step1: Collect Edges ==========\n";
        for (int i = 0; i < edges.size(); ++i)
        {
            const auto &e = edges[i];
            const auto &c = globalCells->at(e.cellIdx);

            // std::cout << "Edge " << i
            //           << " cell=" << e.cellIdx
            //           << " dir=" << e.dir
            //           << " coord=(" << c.coord.x() << "," << c.coord.y() << ")\n";
        }

        if (edges.empty())
            return;

        // =====================================================
        // 2. 分组（dir + 轴）
        // =====================================================
        std::unordered_map<int64_t, std::vector<BoundaryEdge>> groups;

        for (const auto &e : edges)
        {
            const auto &c = globalCells->at(e.cellIdx);

            int keyAxis;

            if (e.dir == 0 || e.dir == 2)
            {
                // horizontal → same Y
                keyAxis = c.coord.y();
            }
            else
            {
                // vertical → same X
                keyAxis = c.coord.x();
            }

            int64_t fullKey = ((int64_t)e.dir << 32) | (uint32_t)keyAxis;
            groups[fullKey].push_back(e);
        }

        // std::cout << "\n========== Step2: Grouping ==========\n";
        for (auto &kv : groups)
        {
            int dir = (int)(kv.first >> 32);
            int axis = (int)(kv.first & 0xffffffff);

            // std::cout << "Group dir=" << dir << " axis=" << axis
            //           << " size=" << kv.second.size() << "\n";
        }

        // =====================================================
        // 3. 每组排序 + 连续性拆分
        // =====================================================
        // std::cout << "\n========== Step3: Build Segments ==========\n";

        for (auto &kv : groups)
        {
            auto &list = kv.second;
            if (list.empty())
                continue;

            int dir = list[0].dir;

            // -------- 排序 --------
            std::sort(list.begin(), list.end(),
                      [&](const BoundaryEdge &a, const BoundaryEdge &b)
                      {
                          const auto &ca = globalCells->at(a.cellIdx);
                          const auto &cb = globalCells->at(b.cellIdx);

                          if (dir == 0 || dir == 2)
                              return ca.coord.x() < cb.coord.x(); // horizontal
                          else
                              return ca.coord.y() < cb.coord.y(); // vertical
                      });

            // -------- debug 排序结果 --------
            // std::cout << "\nSorted group dir=" << dir << ":\n";
            // for (auto &e : list)
            // {
            //     const auto &c = globalCells->at(e.cellIdx);
            //     std::cout << "  cell=" << e.cellIdx
            //               << " coord=(" << c.coord.x() << "," << c.coord.y() << ")\n";
            // }

            // -------- 连续性拆分 --------
            ContourSegment seg;
            seg.dir = dir;

            seg.segments.push_back(list[0]);

            for (int i = 1; i < list.size(); ++i)
            {
                const auto &prev = list[i - 1];
                const auto &cur = list[i];

                const auto &cp = globalCells->at(prev.cellIdx);
                const auto &cc = globalCells->at(cur.cellIdx);

                bool continuous = false;

                if (dir == 0 || dir == 2)
                {
                    // horizontal → y 相同，x 连续
                    continuous = (cc.coord.y() == cp.coord.y()) &&
                                 (cc.coord.x() == cp.coord.x() + 1);
                }
                else
                {
                    // vertical → x 相同，y 连续
                    continuous = (cc.coord.x() == cp.coord.x()) &&
                                 (cc.coord.y() == cp.coord.y() + 1);
                }

                if (!continuous)
                {
                    // ---- 断开，保存当前 segment ----
                    contourSegments.push_back(seg);

                    // std::cout << "  [Split] new segment\n";

                    seg = ContourSegment();
                    seg.dir = dir;
                }

                seg.segments.push_back(cur);
            }

            if (!seg.segments.empty())
                contourSegments.push_back(seg);
        }

        for (auto &seg : contourSegments)
        {
            int dir = seg.dir;

            // Down / Left 需要反转
            if (dir == 1 || dir == 2)
            {
                std::reverse(seg.segments.begin(), seg.segments.end());
            }
        }

        // =====================================================
        // 4. 最终结果打印
        // =====================================================
        // std::cout << "\n========== Final Segments ==========\n";

        // for (int i = 0; i < contourSegments.size(); ++i)
        // {
        //     const auto &seg = contourSegments[i];

        //     std::cout << "Segment " << i
        //               << " dir=" << seg.dir
        //               << " size=" << seg.segments.size() << "\n";

        //     const auto &first = seg.segments.front();
        //     const auto &last = seg.segments.back();

        //     const auto &c0 = globalCells->at(first.cellIdx);
        //     const auto &c1 = globalCells->at(last.cellIdx);

        //     std::cout << "  start coord=(" << c0.coord.x() << "," << c0.coord.y() << ")\n";
        //     std::cout << "  end   coord=(" << c1.coord.x() << "," << c1.coord.y() << ")\n";
        //     std::cout << " startPt=(" << c0.getEdge(first.dir).first.x() << "," << c0.getEdge(first.dir).first.y() << ")\n";
        //     std::cout << "   endPt=(" << c1.getEdge(last.dir).second.x() << "," << c1.getEdge(last.dir).second.y() << ")\n";
        // }
    }

    void CellGroup::buildContour()
    {
        contourPoly.points.clear();

        if (contourSegments.empty() || !globalCells)
            return;

        const float EPS = 1e-6f;
        int N = (int)contourSegments.size();

        struct SegInfo
        {
            Eigen::Vector2f startPt;
            Eigen::Vector2f endPt;
            Eigen::Vector2i startCoord;
            Eigen::Vector2i endCoord;
        };

        std::vector<SegInfo> segs(N);

        // ============================================
        // 1️⃣ 提取 segment 信息
        // ============================================
        for (int i = 0; i < N; ++i)
        {
            auto &seg = contourSegments[i];

            auto &first = seg.segments.front();
            auto &last = seg.segments.back();

            const auto &c0 = globalCells->at(first.cellIdx);
            const auto &c1 = globalCells->at(last.cellIdx);

            auto [p0s, p0e] = c0.getEdge(first.dir);
            auto [p1s, p1e] = c1.getEdge(last.dir);

            segs[i].startPt = p0s;
            segs[i].endPt = p1e;

            segs[i].startCoord = c0.coord;
            segs[i].endCoord = c1.coord;

            // 初始化
            contourSegments[i].startConvex = false;
            contourSegments[i].endConvex = false;
        }

        // ============================================
        // 2️⃣ traversal（严格 end → start）
        // ============================================
        std::vector<bool> used(N, false);
        std::vector<Eigen::Vector2f> pts;

        int start = 0;
        int cur = start;

        used[cur] = true;

        pts.push_back(segs[cur].startPt);
        pts.push_back(segs[cur].endPt);

        Eigen::Vector2f curEnd = segs[cur].endPt;

        // std::cout << "\n=========== Traversal ===========\n";
        // std::cout << "Start seg = " << cur << "\n";

        // ============================================
        // 主循环
        // ============================================
        for (int step = 0; step < N - 1; ++step)
        {
            int next = -1;

            for (int i = 0; i < N; ++i)
            {
                if (used[i] || i == cur)
                    continue;

                //  唯一规则：end → start 精确匹配
                if ((segs[i].startPt - curEnd).squaredNorm() < EPS)
                {
                    next = i;
                    break;
                }
            }

            if (next == -1)
            {
                // std::cout << "[ERROR] cannot find next segment\n";
                break;
            }

            // ========================================
            // convex 判断（你的定义）
            // ========================================
            auto &c0 = segs[cur].endCoord;
            auto &c1 = segs[next].startCoord;

            bool convex = (c0 == c1);

            contourSegments[cur].endConvex = convex;
            contourSegments[next].startConvex = convex;

            // std::cout << "Connect " << cur << " -> " << next
            //           << " convex=" << convex << "\n";

            // ========================================
            // 拼接
            // ========================================
            pts.push_back(segs[next].endPt);

            used[next] = true;
            cur = next;
            curEnd = segs[cur].endPt;
        }

        // ============================================
        // 闭环 convex
        // ============================================
        {
            int first = start;
            int last = cur;

            auto &c0 = segs[last].endCoord;
            auto &c1 = segs[first].startCoord;

            bool convex = (c0 == c1);

            contourSegments[last].endConvex = convex;
            contourSegments[first].startConvex = convex;

            // std::cout << "Connect " << last << " -> " << first
            //           << " (CLOSE LOOP) convex=" << convex << "\n";
        }

        // ============================================
        // 4️⃣ 闭合 polyline
        // ============================================
        if (!pts.empty())
        {
            if ((pts.front() - pts.back()).squaredNorm() > EPS)
                pts.push_back(pts.front());
        }

        contourPoly = geo::Polyline2_t<float>(pts, true);

        // ============================================
        // DEBUG: contour
        // ============================================
        // std::cout << "\n=========== FINAL CONTOUR ===========\n";
        // for (int i = 0; i < pts.size(); ++i)
        // {
        //     std::cout << "P[" << i << "] = ("
        //               << pts[i].x() << ", "
        //               << pts[i].y() << ")\n";
        // }
        // std::cout << "Point count = " << pts.size() << "\n";
        // std::cout << "====================================\n";

        // ============================================
        // DEBUG: segment detail
        // ============================================
        // std::cout << "\n=========== SEGMENT DETAIL ===========\n";

        // for (int i = 0; i < contourSegments.size(); ++i)
        // {
        //     const auto &seg = contourSegments[i];

        //     const auto &first = seg.segments.front();
        //     const auto &last = seg.segments.back();

        //     const auto &c0 = globalCells->at(first.cellIdx);
        //     const auto &c1 = globalCells->at(last.cellIdx);

        //     auto [p0s, p0e] = c0.getEdge(first.dir);
        //     auto [p1s, p1e] = c1.getEdge(last.dir);

        //     std::cout << "Segment " << i << "\n";

        //     std::cout << "  startCoord = ("
        //               << c0.coord.x() << ", "
        //               << c0.coord.y() << ")\n";

        //     std::cout << "  startPt    = ("
        //               << p0s.x() << ", "
        //               << p0s.y() << ")\n";

        //     std::cout << "  startConvex = "
        //               << (seg.startConvex ? "true" : "false") << "\n";

        //     std::cout << "  endCoord   = ("
        //               << c1.coord.x() << ", "
        //               << c1.coord.y() << ")\n";

        //     std::cout << "  endPt      = ("
        //               << p1e.x() << ", "
        //               << p1e.y() << ")\n";

        //     std::cout << "  endConvex  = "
        //               << (seg.endConvex ? "true" : "false") << "\n";

        //     std::cout << "-------------------------------------\n";
        // }

        // std::cout << "=====================================\n";
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
                Eigen::Vector2f center(minX + (ix + 0.5f) * cellSize, minY + (iy + 0.5f) * cellSize);

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
        // 上（0，1）右（1，0）下（0，-1）左（-1，0）
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