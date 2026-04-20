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
        std::reverse(pts.begin(), pts.end());
        contourPoly = geo::Polyline2_t<float>(pts, true);
        // std::cout << "\n=========== CONTOUR POLY DEBUG ===========\n";
        // std::cout << "points count = " << contourPoly.points.size() << "\n";

        // for (size_t i = 0; i < contourPoly.points.size(); ++i)
        // {
        //     const auto &p = contourPoly.points[i];

        //     std::cout << i << ": ("
        //               << p.x() << ", "
        //               << p.y() << ")";

        //     if (i + 1 < contourPoly.points.size())
        //         std::cout << " -> ";

        //     // 每 6 个点换行（避免太长）
        //     if (i % 6 == 5)
        //         std::cout << "\n";
        // }

        // std::cout << "\n==========================================\n";
    }

    void CellGroup::buildMultipleContours()
    {
        contourPolys.clear();

        if (!globalCells || cellIndices.empty())
            return;

        // group 内的 cell
        std::unordered_set<int> groupSet(
            cellIndices.begin(),
            cellIndices.end());

        std::unordered_set<int> visited;

        for (int seed : cellIndices)
        {
            if (visited.count(seed))
                continue;

            // =========================
            // BFS 找一个连通块
            // =========================
            std::vector<int> component;
            std::queue<int> q;

            q.push(seed);
            visited.insert(seed);

            while (!q.empty())
            {
                int cur = q.front();
                q.pop();

                component.push_back(cur);

                const auto &c = globalCells->at(cur);

                for (int d = 0; d < 4; ++d)
                {
                    int nb = c.neighbors[d];

                    // ⭐ 核心逻辑：
                    // 只在 group 内扩展
                    if (nb != -1 &&
                        groupSet.count(nb) &&
                        !visited.count(nb))
                    {
                        visited.insert(nb);
                        q.push(nb);
                    }
                }
            }

            // =========================
            // 用这个 component 建 contour
            // =========================
            CellGroup subGroup(component, globalCells, 0);

            if (!subGroup.contourPoly.points.empty())
            {
                contourPolys.push_back(subGroup.contourPoly);
            }

            std::cout << "[DEBUG] component size = "
                      << component.size() << std::endl;
        }

        std::cout << "[DEBUG] total components = "
                  << contourPolys.size() << std::endl;
    }

    std::vector<std::vector<int>> CellGroup::findRectGroups() const
    {
        std::vector<std::vector<int>> result;

        if (!globalCells || cellIndices.empty())
            return result;

        // ----------------------------
        // 1. 构建 occupancy grid
        // ----------------------------
        std::unordered_map<int64_t, int> coord2idx;

        int minx = INT_MAX, miny = INT_MAX;
        int maxx = INT_MIN, maxy = INT_MIN;

        for (int idx : cellIndices)
        {
            auto &c = (*globalCells)[idx];
            minx = std::min(minx, c.coord.x());
            miny = std::min(miny, c.coord.y());
            maxx = std::max(maxx, c.coord.x());
            maxy = std::max(maxy, c.coord.y());

            coord2idx[encode(c.coord.x(), c.coord.y())] = idx;
        }

        int W = maxx - minx + 1;
        int H = maxy - miny + 1;

        std::vector<std::vector<int>> grid(H, std::vector<int>(W, 0));
        std::vector<std::vector<int>> idGrid(H, std::vector<int>(W, -1));

        for (auto &kv : coord2idx)
        {
            int x = int(kv.first >> 32);
            int y = int(kv.first & 0xffffffff);

            int gx = x - minx;
            int gy = y - miny;

            grid[gy][gx] = 1;
            idGrid[gy][gx] = kv.second;
        }

        // ----------------------------
        // 2. 逐层找最大矩形
        // ----------------------------
        std::vector<std::vector<int>> used = grid;

        while (true)
        {
            std::vector<int> height(W, 0);

            int bestArea = 0;
            int bestL = 0, bestR = 0, bestTop = 0, bestBottom = 0;

            for (int y = 0; y < H; ++y)
            {
                for (int x = 0; x < W; ++x)
                {
                    height[x] = (used[y][x] == 1) ? height[x] + 1 : 0;
                }

                // 单调栈求最大矩形
                std::stack<int> st;
                for (int i = 0; i <= W; ++i)
                {
                    int h = (i == W ? 0 : height[i]);

                    while (!st.empty() && height[st.top()] > h)
                    {
                        int top = st.top();
                        st.pop();

                        int left = st.empty() ? 0 : st.top() + 1;
                        int right = i - 1;

                        int area = height[top] * (right - left + 1);

                        if (area > bestArea)
                        {
                            bestArea = area;
                            bestTop = y;
                            bestBottom = y - height[top] + 1;
                            bestL = left;
                            bestR = right;
                        }
                    }
                    st.push(i);
                }
            }

            if (bestArea == 0)
                break;

            // ----------------------------
            // 3. 收集这个矩形
            // ----------------------------
            std::vector<int> rect;

            for (int y = bestBottom; y <= bestTop; ++y)
            {
                for (int x = bestL; x <= bestR; ++x)
                {
                    rect.push_back(idGrid[y][x]);
                    used[y][x] = 0; // remove
                }
            }

            result.push_back(rect);
        }

        return result;
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

    void CellRegion::swapEdgeCells()
    {
    }

    void CellRegion::mergeSingleCell()
    {
        if (!globalCells || groups.empty())
            return;

        std::vector<int> cellOwner(globalCells->size(), -1);

        // init owner
        for (int gi = 0; gi < groups.size(); ++gi)
        {
            for (int idx : groups[gi].cellIndices)
                cellOwner[idx] = gi;
        }

        bool changed = true;

        while (changed)
        {
            changed = false;

            for (int gi = 0; gi < groups.size(); ++gi)
            {
                auto &grp = groups[gi];

                for (int ci = 0; ci < (int)grp.cellIndices.size(); ++ci)
                {
                    int idx = grp.cellIndices[ci];
                    const auto &cell = (*globalCells)[idx];

                    std::unordered_map<int, int> neighborCount;

                    int selfCount = 0;

                    // ===========================
                    // 统计邻居 group
                    // ===========================
                    for (int d = 0; d < 4; ++d)
                    {
                        int nb = cell.neighbors[d];
                        if (nb < 0)
                            continue;

                        int nid = cellOwner[nb];

                        if (nid == gi)
                            selfCount++;
                        else if (nid >= 0)
                            neighborCount[nid]++;
                    }

                    // ===========================
                    // ⭐ 关键条件：孤岛 / 毛刺
                    // ===========================
                    if (selfCount > 1)
                        continue; // 不是突触

                    if (neighborCount.empty())
                        continue;

                    // ===========================
                    // 找最强邻居 group
                    // ===========================
                    int bestGroup = -1;
                    int bestCnt = -1;

                    for (auto &kv : neighborCount)
                    {
                        if (kv.second > bestCnt)
                        {
                            bestCnt = kv.second;
                            bestGroup = kv.first;
                        }
                    }

                    if (bestGroup < 0 || bestGroup == gi)
                        continue;

                    auto &target = groups[bestGroup];

                    // ===========================
                    // try move
                    // ===========================
                    size_t oldSize = target.contourPoly.points.size();

                    target.cellIndices.push_back(idx);
                    grp.cellIndices.erase(grp.cellIndices.begin() + ci);

                    target.buildContourSegments();
                    target.buildContour();

                    grp.buildContourSegments();
                    grp.buildContour();

                    size_t newSize = target.contourPoly.points.size();

                    // ===========================
                    // accept rule
                    // ===========================
                    bool accept = (newSize <= oldSize + 2);

                    std::cout << "[TRY] cell " << idx
                              << " from " << gi
                              << " -> " << bestGroup
                              << " selfCount=" << selfCount
                              << " old=" << oldSize
                              << " new=" << newSize
                              << " accept=" << accept << "\n";

                    if (accept)
                    {
                        cellOwner[idx] = bestGroup;
                        changed = true;
                        break;
                    }
                    else
                    {
                        // rollback
                        target.cellIndices.pop_back();
                        grp.cellIndices.insert(grp.cellIndices.begin() + ci, idx);

                        target.buildContourSegments();
                        target.buildContour();

                        grp.buildContourSegments();
                        grp.buildContour();
                    }
                }

                if (changed)
                    break;
            }
        }

        // ===========================
        // debug
        // ===========================
        std::cout << "\n===== FINAL GROUPS =====\n";

        for (int gi = 0; gi < groups.size(); ++gi)
        {
            std::cout << "group" << gi << ": ";
            for (auto v : groups[gi].cellIndices)
                std::cout << v << ",";
            std::cout << "\n";
        }
    }

    void CellRegion::pushAdditionalCells()
    {
    }

    void CellRegion::buildContourMeshes(const std::vector<float> &baseHeights, const std::vector<int> &floors)
    {
        if (groups.empty())
            return;

        if (groups.size() != baseHeights.size())
        {
            std::cerr << "[ERROR] baseHeights size should match groups size\n";
            return;
        }

        if (groups.size() != floors.size())
        {
            std::cerr << "[ERROR] floors size should match groups size\n";
            return;
        }

        for (int i = 0; i < groups.size(); ++i)
        {
            const auto &cellGrp = groups[i];
            float height = baseHeights[i];
            int floor = floors[i];
            std::vector<Eigen::Vector3f> pts3d = geo::convertPolyline2To3D(cellGrp.contourPoly, height);
            contourMeshes.emplace_back(pts3d, floor * 4.0f);
        }
    }

    void FloorSystem::build(
        const std::vector<std::vector<int>> &groupIndices,
        const std::vector<float> &baseHeights,
        const std::vector<int> &floors,
        const std::vector<int> &isAffect)
    {
        layers.clear();
        floorMeshes.clear();
        yardMeshes.clear();

        int N = groupIndices.size();

        // =========================
        // 1. h_min
        // =========================
        float h_min = 1e9;
        for (int i = 0; i < N; ++i)
            if (floors[i] > 0)
                h_min = std::min(h_min, baseHeights[i]);

        if (h_min == 1e9)
            h_min = 0.0f;

        auto snap2m = [&](float h)
        {
            int k = std::round((h - h_min) / 2.0f);
            return h_min + k * 2.0f;
        };

        // =========================
        // 2. 建筑层 map（只放 building）
        // =========================
        std::map<float, std::vector<int>> layerCells;

        for (int i = 0; i < N; ++i)
        {
            if (floors[i] <= 0)
                continue;

            float base = snap2m(baseHeights[i]);

            for (int k = 0; k < floors[i]; ++k)
            {
                float h = base + k * 4.0f;

                auto &cells = layerCells[h];
                cells.insert(cells.end(),
                             groupIndices[i].begin(),
                             groupIndices[i].end());
            }
        }

        // =========================
        // 3. 建筑 mesh（关键修复点）
        // =========================
        for (auto &kv : layerCells)
        {
            float h = kv.first;

            // 去重
            std::unordered_set<int> uniq(
                kv.second.begin(),
                kv.second.end());

            std::vector<int> cells(
                uniq.begin(),
                uniq.end());

            std::cout << "\n[Layer] h=" << h
                      << " cells=" << cells.size() << std::endl;

            // ⭐ 多连通处理
            CellGroup tmp(cells, globalCells, 0);
            tmp.buildMultipleContours();

            for (auto &poly : tmp.contourPolys)
            {
                auto pts3d = geo::convertPolyline2To3D(poly, h);

                floorMeshes.emplace_back(pts3d, 4.0f);
            }
        }

        // =========================
        // 4. yard（完全独立）
        // =========================
        for (int i = 0; i < N; ++i)
        {
            if (floors[i] != 0)
                continue;

            float h = (baseHeights[i] < h_min)
                          ? h_min
                          : snap2m(baseHeights[i]);

            CellGroup yardGrp(groupIndices[i], globalCells, 0);

            auto poly = yardGrp.contourPoly;

            auto pts3d = geo::convertPolyline2To3D(poly, h);

            yardMeshes.emplace_back(pts3d, 0.3f);

            std::cout << "[YARD] group " << i
                      << " cells=" << groupIndices[i].size()
                      << " h=" << h << std::endl;
        }
    }
}