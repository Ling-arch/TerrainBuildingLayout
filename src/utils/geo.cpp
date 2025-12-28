#include "geo.h"


namespace geo{
    PointKey makeKey(const Eigen::Vector3f &p, float eps)
    {
        return {
            static_cast<int> (std::round(p.x() / eps)),
            static_cast<int> (std::round(p.y() / eps)),
            static_cast<int> (std::round(p.z() / eps))};
    }

    std::vector<Polyline> buildPolylines(const std::vector<geo::Segment> &segments, float eps)
    {
        using Key = PointKey;

        std::unordered_map<Key, std::vector<Edge>, PointKeyHash> adjacency;

        // ===== 1. 构建端点邻接 =====
        for (int i = 0; i < (int)segments.size(); ++i)
        {
            const auto &s = segments[i];

            adjacency[makeKey(s.p0, eps)].push_back({i, 0});
            adjacency[makeKey(s.p1, eps)].push_back({i, 1});
        }

        std::vector<bool> used(segments.size(), false);
        std::vector<Polyline> polylines;

        // ===== 2. 从端点（度=1）开始，生成开曲线 =====
        for (auto &[key, edges] : adjacency)
        {
            if (edges.size() != 1)
                continue;

            int segId = edges[0].segId;
            if (used[segId])
                continue;

            Polyline pl;
            followChain(segId, edges[0].end,
                        segments, adjacency, used, eps, pl);
            polylines.push_back(std::move(pl));
        }

        // ===== 3. 剩余的是闭合曲线 =====
        for (int i = 0; i < (int)segments.size(); ++i)
        {
            if (used[i])
                continue;

            Polyline pl;
            followChain(i, 0,
                        segments, adjacency, used, eps, pl);
            pl.closed = true;
            polylines.push_back(std::move(pl));
        }

        return polylines;
    }

    void followChain(
        int startSeg,
        int startEnd,
        const std::vector<geo::Segment> &segments,
        const std::unordered_map<PointKey,std::vector<Edge>, PointKeyHash> &adj,
        std::vector<bool> &used,
        float eps,
        Polyline &out)
    {
        int currSeg = startSeg;
        int currEnd = startEnd;

        Eigen::Vector3f currPt =
            (currEnd == 0) ? segments[currSeg].p0
                           : segments[currSeg].p1;

        out.points.push_back(currPt);

        while (true)
        {
            used[currSeg] = true;

            // 走到 segment 的另一端
            Eigen::Vector3f nextPt =
                (currEnd == 0) ? segments[currSeg].p1
                               : segments[currSeg].p0;

            out.points.push_back(nextPt);

            PointKey key = makeKey(nextPt, eps);
            auto it = adj.find(key);
            if (it == adj.end())
                break;

            // 寻找下一条未使用的 segment
            int nextSeg = -1;
            int nextEnd = -1;

            for (const auto &e : it->second)
            {
                if (e.segId != currSeg && !used[e.segId])
                {
                    nextSeg = e.segId;
                    nextEnd = e.end;
                    break;
                }
            }

            if (nextSeg == -1)
                break;

            currSeg = nextSeg;
            currEnd = nextEnd;
        }
    }
}