
#include "SCARoadGenerator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_set>

namespace SCARoad
{

    void SCANetwork::initPaths()
    {
        // 每个 seed 初始化为一条 path
        nodeToPath.resize(nodes.size(), -1);
        for (int i = 0; i < nodes.size(); ++i)
        {
            paths.push_back({i});
            nodeToPath[i] = i;
        }
    }

    void SCANetwork::buildKDTree()
    {
        nodePoints.clear();
        for (auto &n : nodes)
            nodePoints.push_back(n.position);

        cloud = std::make_unique<PointCloud2D>(nodePoints);

        tree = std::make_unique<KDTree>(2, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

        tree->buildIndex();
    }

    std::vector<int> SCANetwork::getNodesInRadius(const Eigen::Vector2f &pos, float radius) const
    {
        std::vector<int> result;

        float query[2] = {pos.x(), pos.y()};
        float r2 = radius * radius;

        std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
        nanoflann::SearchParameters params;

        tree->radiusSearch(query, r2, matches, params);

        for (auto &m : matches)
            result.push_back(m.first);

        return result;
    }

    int SCANetwork::getClosestNode(int attractorID)
    {
        auto &a = attractors[attractorID];

        auto nearby = getNodesInRadius(a.position, attractionDist);

        int closest = -1;
        float record = attractionDist;

        for (int nid : nearby)
        {
            float d = (nodes[nid].position - a.position).norm();

            if (d < killDist)
            {
                a.reached = true;
                return -1;
            }
            else if (d < record)
            {
                record = d;
                closest = nid;
            }
        }

        return closest;
    }

    std::vector<int> SCANetwork::getRelativeNeighbors(int attractorID) const
    {
        auto &a = attractors[attractorID];

        auto nearby = getNodesInRadius(a.position, attractionDist);

        std::vector<int> result;

        for (int p0 : nearby)
        {
            bool fail = false;

            auto v0 = nodes[p0].position - a.position;

            for (int p1 : nearby)
            {
                if (p0 == p1)
                    continue;

                auto v1 = nodes[p1].position - a.position;

                if (v1.squaredNorm() > v0.squaredNorm())
                    continue;

                auto p0p1 = nodes[p1].position - nodes[p0].position;

                if (v0.squaredNorm() > p0p1.squaredNorm())
                {
                    fail = true;
                    break;
                }
            }

            if (!fail)
                result.push_back(p0);
        }

        return result;
    }

    std::vector<int> SCANetwork::getRelativeNeighbors(Eigen::Vector2f pos, float radius) const
    {
        auto nearby = getNodesInRadius(pos, radius);

        std::vector<int> result;

        for (int p0 : nearby)
        {
            bool fail = false;

            auto v0 = nodes[p0].position - pos;

            for (int p1 : nearby)
            {
                if (p0 == p1)
                    continue;

                auto v1 = nodes[p1].position - pos;

                if (v1.squaredNorm() > v0.squaredNorm())
                    continue;

                auto p0p1 = nodes[p1].position - nodes[p0].position;

                if (v0.squaredNorm() > p0p1.squaredNorm())
                {
                    fail = true;
                    break;
                }
            }

            if (!fail)
                result.push_back(p0);
        }

        return result;
    }

    Eigen::Vector2f SCANetwork::getAverageDirection(int nodeID) const
    {
        auto &node = nodes[nodeID];

        Eigen::Vector2f dir(0, 0);

        for (int aid : node.influencedBy)
        {
            auto v = (attractors[aid].position - node.position).normalized();
            dir += v;
        }

        dir += Eigen::Vector2f(
            (rand() / (float)RAND_MAX - 0.5f) * 0.2f,
            (rand() / (float)RAND_MAX - 0.5f) * 0.2f);

        if (dir.norm() > 1e-6f)
            dir.normalize();

        return dir;
    }

    void SCANetwork::update(bool &growthStopped)
    {
        growthStopped = false;

        const float STEP_SIZE = 5.0f;
        const float CONNECT_DIST = 20.0f;
        const float CONNECT_DIST2 = CONNECT_DIST * CONNECT_DIST;

        const int MIN_BRANCH_GAP = 7; //  分叉间隔
        const int SAME_PATH_GAP = 12;  //  同path loop间隔

        std::cout << "\n========== [SCA UPDATE BEGIN] ==========\n";

        // ============================
        // 1. Reset
        // ============================
        for (auto &a : attractors)
            a.resetFrame();

        for (auto &n : nodes)
            n.influencedBy.clear();

        // ============================
        // 2. Attractor → Node
        // ============================
        for (int i = 0; i < attractors.size(); ++i)
        {
            auto &a = attractors[i];

            auto neighbors = getRelativeNeighbors(i);
            auto killNodes = getNodesInRadius(a.position, killDist);

            for (int n : neighbors)
            {
                if (std::find(killNodes.begin(), killNodes.end(), n) == killNodes.end())
                {
                    nodes[n].influencedBy.push_back(i);
                }
            }
        }

        // ============================
        // 3. Grow Nodes
        // ============================
        std::vector<SCANode> newNodes;
        std::vector<int> newParents;

        for (int i = 0; i < nodes.size(); ++i)
        {
            auto &node = nodes[i];

            if (node.influencedBy.empty())
                continue;

            if (node.links.size() >= 4)
                continue;

            Eigen::Vector2f dir = getAverageDirection(i);
            if (dir.norm() < 1e-6f)
                continue;

            //分叉限制
            bool allowGrow = true;

            if (node.paths.size() == 1)
            {
                int pid = node.paths[0];
                auto &path = paths[pid];

                auto it = std::find(path.begin(), path.end(), i);
                int idx = std::distance(path.begin(), it);

                bool isEnd = (idx == 0 || idx == path.size() - 1);

                if (!isEnd)
                {
                    int dist = findNearestBranchDist(pid, i);

                    if (dist < MIN_BRANCH_GAP)
                    {
                        allowGrow = false;

                        std::cout << "  [Skip Branch] Node[" << i
                                  << "] too close to branch\n";
                    }
                }
            }

            if (!allowGrow)
                continue;

            Eigen::Vector2f newPos = node.position + dir * STEP_SIZE;

            SCANode newNode;
            newNode.position = newPos;
            newNode.parent = i;

            newNodes.push_back(newNode);
            newParents.push_back(i);

            node.isTip = false;
        }

        if (newNodes.empty())
        {
            growthStopped = true;
            return;
        }

        // ============================
        // 4. Merge + Links
        // ============================
        int oldSize = nodes.size();
        nodes.insert(nodes.end(), newNodes.begin(), newNodes.end());

        for (int i = 0; i < newNodes.size(); ++i)
        {
            int newID = oldSize + i;
            int parent = newParents[i];

            nodes[newID].links.push_back(parent);
            nodes[parent].links.push_back(newID);
        }

        // ============================
        // 5. Path 更新
        // ============================
        for (int i = 0; i < newNodes.size(); ++i)
        {
            int newID = oldSize + i;
            int parent = newParents[i];

            auto &parentNode = nodes[parent];
            auto &newNode = nodes[newID];

            bool attached = false;

            for (int pid : parentNode.paths)
            {
                auto &path = paths[pid];

                auto it = std::find(path.begin(), path.end(), parent);
                int idx = std::distance(path.begin(), it);

                bool isHead = (idx == 0);
                bool isTail = (idx == path.size() - 1);

                if (isTail)
                {
                    path.push_back(newID);
                    newNode.paths.push_back(pid);
                    attached = true;
                    break;
                }
                else if (isHead)
                {
                    path.insert(path.begin(), newID);
                    newNode.paths.push_back(pid);
                    attached = true;
                    break;
                }
            }

            if (!attached)
            {
                int newPID = paths.size();
                paths.push_back({parent, newID});

                parentNode.paths.push_back(newPID);
                newNode.paths.push_back(newPID);
            }
        }

        // ============================
        // 6. Loop（最终正确版本）
        // ============================
        buildKDTree();

        std::unordered_set<int> usedNewNodes; // 防止重复连接

        for (int i = 0; i < newNodes.size(); ++i)
        {
            int newID = oldSize + i;

            if (usedNewNodes.count(newID))
                continue;

            auto &newNode = nodes[newID];
            int parent = newParents[i];

            // ============================
            // 6.1 parent邻域屏蔽
            // ============================
            std::unordered_set<int> forbidden;

            for (int pid : nodes[parent].paths)
            {
                auto &path = paths[pid];

                auto it = std::find(path.begin(), path.end(), parent);
                int idx = std::distance(path.begin(), it);

                for (int k = -SAME_PATH_GAP; k <= SAME_PATH_GAP; ++k)
                {
                    int id = idx + k;
                    if (id >= 0 && id < path.size())
                        forbidden.insert(path[id]);
                }
            }
            

            auto nearby = getRelativeNeighbors(newNode.position, CONNECT_DIST);

            int best = -1;
            float bestScore = -1;

            for (int nid : nearby)
            {
                if (nid == parent)
                    continue;

                if (forbidden.count(nid))
                    continue;

                if (nid >= oldSize && usedNewNodes.count(nid))
                    continue;

                float d2 = (nodes[nid].position - newNode.position).squaredNorm();
                if (d2 > CONNECT_DIST2)
                    continue;

                // =====================
                // path gap限制
                // =====================
                if (tooCloseInSamePath(newID, nid, SAME_PATH_GAP))
                    continue;

                bool sharePath = false;
                for (int p1 : newNode.paths)
                    for (int p2 : nodes[nid].paths)
                        if (p1 == p2)
                            sharePath = true;

                if (sharePath && tooCloseInSamePath(newID, nid, MIN_BRANCH_GAP))
                    continue;

                // =====================
                // score
                // =====================
                Eigen::Vector2f dir1 =
                    (newNode.position - nodes[parent].position).normalized();

                Eigen::Vector2f dir2 =
                    (nodes[nid].position - newNode.position).normalized();

                float angleScore = 1.0f - fabs(dir1.dot(dir2));
                float score = angleScore / (1.0f + d2 * 0.1f);

                if (score > bestScore)
                {
                    bestScore = score;
                    best = nid;
                }
            }

            if (best < 0)
                continue;

            auto &extraNode = nodes[best];

            // ============================
            // 十字路口限制
            // ============================
            if (extraNode.links.size() >= 4)
                continue;

            // ============================
            // 建立 link
            // ============================
            nodes[newID].links.push_back(best);
            nodes[best].links.push_back(newID);

            std::cout << "  [Loop] " << newID << " <-> " << best << "\n";

            // usedNewNodes.insert(newID);
            if (best >= oldSize)
                usedNewNodes.insert(best);

            // ============================
            // Path 修复（核心）
            // ============================

            // 👉 newNode 当前所在 path（第5步保证只有1个）
            int curPID = newNode.paths[0];
            auto &curPath = paths[curPID];

            bool newAtHead = (curPath.front() == newID);
            bool newAtTail = (curPath.back() == newID);

            // ============================
            // 情况1：extraNode 是端点（link=1, path=1）
            // ============================
            if (extraNode.links.size() == 1 && extraNode.paths.size() == 1)
            {
                int pid = extraNode.paths[0];
                auto &path = paths[pid];

                if (path.front() == best)
                {
                    if (newAtTail)
                        curPath.insert(curPath.end(), path.begin(), path.end());
                    else
                        curPath.insert(curPath.begin(), path.begin(), path.end());
                }
                else if (path.back() == best)
                {
                    if (newAtTail)
                        curPath.insert(curPath.end(), path.begin(), path.end());
                    else
                        curPath.insert(curPath.begin(), path.begin(), path.end());
                }

                for (int nid : path)
                    nodes[nid].paths.push_back(curPID);

                std::cout << "    [Merge Path - endpoint]\n";
            }

            // ============================
            // 情况2：extraNode 是中间点（link=2）
            // ============================
            else if (extraNode.links.size() == 2 && extraNode.paths.size() == 1)
            {
                if (newAtTail)
                    curPath.push_back(best);
                else
                    curPath.insert(curPath.begin(), best);

                nodes[best].paths.push_back(curPID);

                std::cout << "    [Attach Mid Node]\n";
            }

            // ============================
            // 情况3：extraNode 是分叉点（link=3, path=2）
            // ============================
            else if (extraNode.links.size() == 3 && extraNode.paths.size() == 2)
            {
                for (int pid : extraNode.paths)
                {
                    auto &path = paths[pid];

                    if (path.front() == best || path.back() == best)
                    {
                        if (newAtTail)
                            curPath.insert(curPath.end(), path.begin(), path.end());
                        else
                            curPath.insert(curPath.begin(), path.begin(), path.end());

                        for (int nid : path)
                            nodes[nid].paths.push_back(curPID);

                        std::cout << "    [Merge Branch Path]\n";
                        break;
                    }
                }
            }

            // ============================
            // 情况4：extraNode 是十字路口（跳过）
            // ============================
        }

        // ============================
        // 7. Remove Attractors
        // ============================
        attractors.erase(
            std::remove_if(
                attractors.begin(),
                attractors.end(),
                [&](Attractor &a)
                {
                    for (auto &n : nodes)
                    {
                        if ((n.position - a.position).norm() < killDist)
                            return true;
                    }
                    return false;
                }),
            attractors.end());

        std::cout << "========== [SCA UPDATE END] ==========\n";
    }

    int SCANetwork::findNearestBranchDist(int pathID, int nodeID) const
    {
        const auto &path = paths[pathID];

        int idx = -1;
        for (int i = 0; i < path.size(); ++i)
        {
            if (path[i] == nodeID)
            {
                idx = i;
                break;
            }
        }

        if (idx < 0)
            return INT_MAX;

        int bestDist = INT_MAX;

        for (int i = 0; i < path.size(); ++i)
        {
            int nid = path[i];

            if (nodes[nid].paths.size() >= 2) // 分叉点
            {
                int d = abs(i - idx);
                if (d > 0)
                    bestDist = std::min(bestDist, d);
            }
        }

        return bestDist;
    }

    // 判断 a、b 是否在同一 path 上且距离小于 minStep
    bool SCANetwork::tooCloseInSamePath(int a, int b, int minStep) const
    {
        for (int pidA : nodes[a].paths)
        {
            for (int pidB : nodes[b].paths)
            {
                if (pidA != pidB)
                    continue;

                const auto &path = paths[pidA];

                int ia = -1, ib = -1;

                for (int i = 0; i < path.size(); ++i)
                {
                    if (path[i] == a)
                        ia = i;
                    if (path[i] == b)
                        ib = i;
                }

                if (ia >= 0 && ib >= 0)
                {
                    if (abs(ia - ib) < minStep)
                        return true;
                }
            }
        }
        return false;
    }

    std::vector<std::vector<Eigen::Vector2f>> SCANetwork::extractRoads() const
    {
        std::vector<std::vector<Eigen::Vector2f>> roads;

        for(const auto &path : paths)
        {
            std::vector<Eigen::Vector2f> road;

            for (int nid : path)
                road.push_back(nodes[nid].position);

            roads.push_back(road);
        }
        // if (nodes.empty())
        //     return roads;

        // // ⭐ 防止重复边（无向图）
        // std::unordered_set<long long> visited;

        // auto edgeKey = [](int a, int b)
        // {
        //     if (a > b)
        //         std::swap(a, b);
        //     return (static_cast<long long>(a) << 32) | b;
        // };

        // // ============================
        // // 1. parent edges（树边）
        // // ============================
        // for (int i = 0; i < nodes.size(); ++i)
        // {
        //     int p = nodes[i].parent;

        //     if (p >= 0)
        //     {
        //         long long key = edgeKey(i, p);

        //         if (visited.count(key) == 0)
        //         {
        //             visited.insert(key);

        //             roads.push_back({nodes[i].position,
        //                              nodes[p].position});
        //         }
        //     }
        // }

        // // ============================
        // // 2. extra links（闭环边）
        // // ============================
        // for (int i = 0; i < nodes.size(); ++i)
        // {
        //     for (int j : nodes[i].extraLinks)
        //     {
        //         long long key = edgeKey(i, j);

        //         if (visited.count(key) == 0)
        //         {
        //             visited.insert(key);

        //             roads.push_back({nodes[i].position,
        //                              nodes[j].position});
        //         }
        //     }
        // }

        return roads;
    }

    Eigen::Vector2f SCANetwork::clampDirection(
        const Eigen::Vector2f &parentDir,
        const Eigen::Vector2f &newDir) const
    {
        float dot = std::clamp(parentDir.dot(newDir), -1.f, 1.f);
        float angle = std::acos(dot);

        float minA = minBranchAngle * Litten_M_PI / 180.f;
        float maxA = maxBranchAngle * Litten_M_PI / 180.f;

        if (angle < minA || angle > maxA)
        {
            float target = (angle < minA) ? minA : maxA;

            float sign = (parentDir.x() * newDir.y() - parentDir.y() * newDir.x()) > 0 ? 1.f : -1.f;

            float c = std::cos(target);
            float s = std::sin(target);

            Eigen::Vector2f rotated(
                parentDir.x() * c - sign * parentDir.y() * s,
                parentDir.x() * sign * s + parentDir.y() * c);

            return rotated.normalized();
        }

        return newDir;
    }

    void SCANetwork::drawNodesWithIndices(render::Renderer3D render) const
    {
        std::vector<Eigen::Vector2f> nodePoints;

        for (auto &n : nodes)
            nodePoints.push_back(n.position);
        render.draw_index_fonts(nodePoints, render.ptData.size, render.ptData.color);
    }

    std::vector<Attractor> getRandomAttractors(int num, float width, float height, std::vector<Eigen::Vector2f> &pos, const Eigen::Vector2f &center)
    {
        pos.clear();
        std::vector<Attractor> attractors;

        AttractorSettings settings;

        float halfW = width * 0.5f;
        float halfH = height * 0.5f;

        for (int i = 0; i < num; i++)
        {
            float x = randomFloat(-halfW, halfW) + center.x();
            float y = randomFloat(-halfH, halfH) + center.y();

            Eigen::Vector2f p(x, y);

            pos.push_back(p);
            attractors.emplace_back(p, settings);
        }

        return attractors;
    }

    void SCANetwork::debugPrintRelativeNeighbors(int nodeIndex, float radius) const
    {
        if (nodeIndex < 0 || nodeIndex >= nodes.size())
        {
            std::cout << "[Error] Invalid node index\n";
            return;
        }

        const auto &pos = nodes[nodeIndex].position;

        std::cout << "\n===== [Relative Neighbors Debug] =====\n";
        std::cout << "Node[" << nodeIndex << "] Pos: ("
                  << pos.x() << ", " << pos.y() << ")\n";
        std::cout << "Radius: " << radius << "\n";

        // 1️⃣ 所有邻域节点
        auto nearby = getNodesInRadius(pos, radius);

        std::cout << "\n[Nearby Nodes]\n";
        for (int nid : nearby)
        {
            const auto &p = nodes[nid].position;
            float dist = (p - pos).norm();

            std::cout << "  Node[" << nid << "] ("
                      << p.x() << ", " << p.y() << ") "
                      << "Dist=" << dist << "\n";
        }

        // 2️⃣ Relative Neighbor
        auto relative = getRelativeNeighbors(pos, radius);

        std::cout << "\n[Relative Neighbors]\n";
        for (int nid : relative)
        {
            const auto &p = nodes[nid].position;
            float dist = (p - pos).norm();

            std::cout << "  Node[" << nid << "] ("
                      << p.x() << ", " << p.y() << ") "
                      << "Dist=" << dist << "\n";
        }

        std::cout << "===== END =====\n";
    }

    void SCANetwork::debugPrintNodePaths(int nodeIndex) const
    {
        if (nodeIndex < 0 || nodeIndex >= nodes.size())
        {
            std::cout << "[Error] Invalid node index\n";
            return;
        }

        const auto &node = nodes[nodeIndex];

        std::cout << "\n===== [Node Path Debug] =====\n";
        std::cout << "Node[" << nodeIndex << "] Pos: ("
                  << node.position.x() << ", "
                  << node.position.y() << ")\n";

        std::cout << "Path Count: " << node.paths.size() << "\n";

        for (int pid : node.paths)
        {
            if (pid < 0 || pid >= paths.size())
            {
                std::cout << "  [Invalid Path ID: " << pid << "]\n";
                continue;
            }

            const auto &path = paths[pid];

            std::cout << "  Path[" << pid << "] Size: "
                      << path.size() << "\n";

            std::cout << "    Nodes: ";

            for (int nid : path)
            {
                const auto &p = nodes[nid].position;

                std::cout << nid << "("
                          << p.x() << ", "
                          << p.y() << ") ";
            }

            std::cout << "\n";
        }

        std::cout << "===== END =====\n";
    }

    void SCANetwork::debugPrintNodeLinks(int nodeIndex) const
    {
        if (nodeIndex < 0 || nodeIndex >= nodes.size())
            return;

        const auto &node = nodes[nodeIndex];

        std::cout << "\n===== [Node Link Debug] =====\n";
        std::cout << "Node[" << nodeIndex << "] ("
                  << node.position.x() << ", "
                  << node.position.y() << ")\n";

        std::cout << "Links (" << node.links.size() << "):\n";

        for (int nid : node.links)
        {
            const auto &p = nodes[nid].position;

            std::cout << "  -> Node[" << nid << "] ("
                      << p.x() << ", "
                      << p.y() << ")\n";
        }

        std::cout << "===== END =====\n";
    }

    void SCANetwork::debugConnectNodes(int nodeIndex, float CONNECT_DIST) const
    {
        const auto &newNode = nodes[nodeIndex];
        int parent = newNode.parent;

        const int SAME_PATH_GAP = 12;
        const int MIN_BRANCH_GAP = 18;
        const float CONNECT_DIST2 = CONNECT_DIST * CONNECT_DIST;

        std::cout << "\n========== [DEBUG CONNECT NODE] ==========\n";

        std::cout << "NewNode[" << nodeIndex << "] Pos: ("
                  << newNode.position.x() << ", "
                  << newNode.position.y() << ")\n";

        std::cout << "Parent[" << parent << "] Pos: ("
                  << nodes[parent].position.x() << ", "
                  << nodes[parent].position.y() << ")\n";

        // ============================
        // 1. forbidden 区域
        // ============================
        std::unordered_set<int> forbidden;

        std::cout << "\n[Step1] Forbidden Nodes (parent neighborhood):\n";

        for (int pid : nodes[parent].paths)
        {
            const auto &path = paths[pid];

            auto it = std::find(path.begin(), path.end(), parent);
            int idx = std::distance(path.begin(), it);

            std::cout << "  Path[" << pid << "] idx=" << idx << ":\n";

            for (int k = -SAME_PATH_GAP; k <= SAME_PATH_GAP; ++k)
            {
                int id = idx + k;
                if (id >= 0 && id < path.size())
                {
                    int nid = path[id];
                    forbidden.insert(nid);

                    const auto &p = nodes[nid].position;

                    std::cout << "    forbid Node[" << nid << "] ("
                              << p.x() << ", " << p.y() << ")\n";
                }
            }
        }

        // ============================
        // 2. 获取候选
        // ============================
        auto nearby = getNodesInRadius(newNode.position, CONNECT_DIST);
        // auto nearby = getRelativeNeighbors(newNode.position, CONNECT_DIST);

        std::cout << "\n[Step2] Nearby Candidates: " << nearby.size() << "\n";

        int best = -1;
        float bestScore = -1;

        for (int nid : nearby)
        {
            const auto &candidate = nodes[nid];

            std::cout << "\n  Checking Node[" << nid << "] ("
                      << candidate.position.x() << ", "
                      << candidate.position.y() << ")\n";

            if (nid == parent)
            {
                std::cout << "    Skip: parent\n";
                continue;
            }

            if (forbidden.count(nid))
            {
                std::cout << "    Skip: forbidden zone\n";
                continue;
            }

            float d2 = (candidate.position - newNode.position).squaredNorm();

            std::cout << "    Dist=" << std::sqrt(d2) << "\n";

            if (d2 > CONNECT_DIST2)
            {
                std::cout << "    Skip: too far\n";
                continue;
            }

            // ===== Case1 同 path =====
            if (tooCloseInSamePath(nodeIndex, nid, SAME_PATH_GAP))
            {
                std::cout << "    Skip: same path too close\n";
                continue;
            }

            // ===== Case2 共享 path =====
            bool sharePath = false;

            for (int p1 : newNode.paths)
                for (int p2 : nodes[nid].paths)
                    if (p1 == p2)
                        sharePath = true;

            if (sharePath)
            {
                std::cout << "    Shared Path detected\n";

                if (tooCloseInSamePath(nodeIndex, nid, MIN_BRANCH_GAP))
                {
                    std::cout << "    Skip: branch gap too small\n";
                    continue;
                }
            }

            // ===== Score =====
            Eigen::Vector2f dir1 =
                (newNode.position - nodes[parent].position).normalized();

            Eigen::Vector2f dir2 =
                (candidate.position - newNode.position).normalized();

            float dot = fabs(dir1.dot(dir2));
            float angleScore = 1.0f - dot;
            float score = angleScore / (1.0f + d2 * 0.1f);

            std::cout << "    AngleScore=" << angleScore
                      << " Score=" << score << "\n";

            if (score > bestScore)
            {
                bestScore = score;
                best = nid;
            }
        }

        // ============================
        // 3. 最终选择
        // ============================
        if (best < 0)
        {
            std::cout << "\n[Result] No valid connection found\n";
            return;
        }

        const auto &extraNode = nodes[best];

        std::cout << "\n[Result] Selected Node[" << best << "] ("
                  << extraNode.position.x() << ", "
                  << extraNode.position.y() << ") "
                  << "Score=" << bestScore << "\n";

        // ============================
        // 十字路口限制
        // ============================
        if (extraNode.links.size() >= 4)
        {
            std::cout << "  [Reject] Crossroad limit (links="
                      << extraNode.links.size() << ")\n";
            return;
        }

        // ============================
        // 当前 path 信息
        // ============================
        int curPID = newNode.paths[0];
        const auto &curPath = paths[curPID];

        bool newAtHead = (curPath.front() == nodeIndex);
        bool newAtTail = (curPath.back() == nodeIndex);

        std::cout << "\n[Current Path] ID=" << curPID
                  << " Head=" << newAtHead
                  << " Tail=" << newAtTail << "\n";

        // ============================
        // Case 1: endpoint
        // ============================
        if (extraNode.links.size() == 1 && extraNode.paths.size() == 1)
        {
            int pid = extraNode.paths[0];
            const auto &path = paths[pid];

            std::cout << "[Case1] Endpoint Node\n";
            std::cout << "  Path[" << pid << "] size=" << path.size() << "\n";

            if (path.front() == best)
            {
                std::cout << "  -> extraNode is PATH HEAD\n";

                if (newAtTail)
                    std::cout << "  -> Merge: curTail -> extraHead\n";
                else
                    std::cout << "  -> Merge: curHead -> extraHead\n";
            }
            else if (path.back() == best)
            {
                std::cout << "  -> extraNode is PATH TAIL\n";

                if (newAtTail)
                    std::cout << "  -> Merge: curTail -> extraTail\n";
                else
                    std::cout << "  -> Merge: curHead -> extraTail\n";
            }
        }

        // ============================
        // Case 2: 中间点
        // ============================
        else if (extraNode.links.size() == 2 && extraNode.paths.size() == 1)
        {
            std::cout << "[Case2] Middle Node\n";

            if (newAtTail)
                std::cout << "  -> Attach at tail\n";
            else
                std::cout << "  -> Attach at head\n";
        }

        // ============================
        // Case 3: 分叉点
        // ============================
        else if (extraNode.links.size() == 3 && extraNode.paths.size() == 2)
        {
            std::cout << "[Case3] Branch Node\n";

            for (int pid : extraNode.paths)
            {
                const auto &path = paths[pid];

                if (path.front() == best || path.back() == best)
                {
                    std::cout << "  -> Found endpoint path: " << pid << "\n";

                    if (newAtTail)
                        std::cout << "  -> Merge branch at tail\n";
                    else
                        std::cout << "  -> Merge branch at head\n";
                }
                else
                {
                    std::cout << "  -> Skip middle path: " << pid << "\n";
                }
            }
        }

        std::cout << "========== [DEBUG END] ==========\n";
    }
}