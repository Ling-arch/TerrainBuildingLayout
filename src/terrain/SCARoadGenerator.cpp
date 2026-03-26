
#include "SCARoadGenerator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_set>

namespace SCARoad
{

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
        // ============================
        // 参数
        // ============================
        const int MAX_NODES = 200000;
        const float STEP_SIZE = 5.0f;
        const float MIN_DIST2 = 4.0f;
        const float CONNECT_DIST = 8.0f;
        const float CONNECT_DIST2 = CONNECT_DIST * CONNECT_DIST;

        if (nodes.size() > MAX_NODES)
        {
            std::cout << "[SCA] Node limit reached\n";
            return;
        }

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

            if (venationType == VenationType::Open)
            {
                int closest = getClosestNode(i);

                if (closest >= 0)
                {
                    nodes[closest].influencedBy.push_back(i);
                    a.influencingNodes = {closest};
                }
            }
            else
            {
                auto neighbors = getRelativeNeighbors(i);
                auto killNodes = getNodesInRadius(a.position, killDist);

                std::vector<int> growNodes;

                for (int n : neighbors)
                {
                    if (std::find(killNodes.begin(), killNodes.end(), n) == killNodes.end())
                        growNodes.push_back(n);
                }

                a.influencingNodes = neighbors;

                if (!growNodes.empty())
                {
                    a.fresh = false;

                    for (int n : growNodes)
                        nodes[n].influencedBy.push_back(i);
                }
            }
        }

        // ============================
        // 3. Grow Nodes
        // ============================
        std::vector<SCANode> newNodes;
        newNodes.reserve(nodes.size());

        for (int i = 0; i < nodes.size(); ++i)
        {
            auto &node = nodes[i];

            if (node.influencedBy.empty())
                continue;

            Eigen::Vector2f dir = getAverageDirection(i);

            if (dir.norm() < 1e-6f)
                continue;

            Eigen::Vector2f newPos = node.position + dir * STEP_SIZE;

            // ============================
            // 3.1 去重
            // ============================
            // bool tooClose = false;

            // for (const auto &n : nodes)
            // {
            //     if ((n.position - newPos).squaredNorm() < MIN_DIST2)
            //     {
            //         tooClose = true;
            //         break;
            //     }
            // }

            // if (!tooClose)
            // {
            //     for (const auto &n : newNodes)
            //     {
            //         if ((n.position - newPos).squaredNorm() < MIN_DIST2)
            //         {
            //             tooClose = true;
            //             break;
            //         }
            //     }
            // }

            // if (tooClose)
            //     continue;

            // ============================
            // 3.2 创建新节点
            // ============================
            SCANode newNode;
            newNode.position = newPos;
            newNode.parent = i;
            newNode.isTip = true;

            newNodes.push_back(newNode);

            node.isTip = false;
        }

        if (newNodes.empty())
        {
            std::cout << "[SCA] Growth stopped\n";
            growthStopped = true;
            return;
        }

        // ============================
        // 4. 合并 nodes
        // ============================
        int oldSize = nodes.size();
        nodes.insert(nodes.end(), newNodes.begin(), newNodes.end());

        // ============================
        // ⭐ 5. 关键：闭环连接（真正生效版）
        // ============================
        buildKDTree();

        for (int i = 0; i < newNodes.size(); ++i)
        {
            int newID = oldSize + i;
            auto &newNode = nodes[newID];

            auto nearby = getNodesInRadius(newNode.position, CONNECT_DIST);

            int bestOld = -1;
            float bestOldScore = -1;

            int bestNew = -1;
            float bestNewScore = -1;

            for (int nid : nearby)
            {
                if (nid == newNode.parent)
                    continue;

                float d2 = (nodes[nid].position - newNode.position).squaredNorm();
                if (d2 > CONNECT_DIST2)
                    continue;

                // ============================
                // ⭐ 角度评估（关键）
                // ============================

                Eigen::Vector2f dirNew =
                    (newNode.position - nodes[newNode.parent].position).normalized();

                Eigen::Vector2f dirCandidate =
                    (nodes[nid].position - newNode.position).normalized();

                float dot = fabs(dirNew.dot(dirCandidate));

                // 👉 理想：接近垂直（论文思想）
                float angleScore = 1.0f - dot;

                float score = angleScore / (1.0f + d2 * 0.1f);

                // ============================
                // ⭐ 分类：老节点 vs 新节点
                // ============================

                if (nid < oldSize)
                {
                    if (score > bestOldScore)
                    {
                        bestOldScore = score;
                        bestOld = nid;
                    }
                }
                else
                {
                    if (score > bestNewScore)
                    {
                        bestNewScore = score;
                        bestNew = nid;
                    }
                }
            }

            // ============================
            // ⭐ 优先连接老节点（形成 loop）
            // ============================

            if (bestOld >= 0 && bestOldScore > 0.2f)
            {
                nodes[newID].extraLinks.push_back(bestOld);
                nodes[bestOld].extraLinks.push_back(newID);
            }
            // ============================
            // ⭐ 次选：新节点（防止断裂）
            // ============================
            else if (bestNew >= 0 && bestNewScore > 0.3f)
            {
                nodes[newID].extraLinks.push_back(bestNew);
                nodes[bestNew].extraLinks.push_back(newID);
            }
        }

        // ============================
        // 6. 删除 attractors
        // ============================
        attractors.erase(
            std::remove_if(
                attractors.begin(),
                attractors.end(),
                [&](Attractor &a)
                {
                    if (venationType == VenationType::Open)
                        return a.reached;

                    if (!a.influencingNodes.empty() && !a.fresh)
                    {
                        bool allReached = true;

                        for (int nid : a.influencingNodes)
                        {
                            if ((nodes[nid].position - a.position).norm() > killDist)
                            {
                                allReached = false;
                                break;
                            }
                        }

                        return allReached;
                    }

                    return false;
                }),
            attractors.end());

        // ============================
        // 7. rebuild KDTree
        // ============================
        //  buildKDTree();
    }

    std::vector<std::vector<Eigen::Vector2f>> SCANetwork::extractRoads() const
    {
        std::vector<std::vector<Eigen::Vector2f>> roads;

        if (nodes.empty())
            return roads;

        // ⭐ 防止重复边（无向图）
        std::unordered_set<long long> visited;

        auto edgeKey = [](int a, int b)
        {
            if (a > b)
                std::swap(a, b);
            return (static_cast<long long>(a) << 32) | b;
        };

        // ============================
        // 1. parent edges（树边）
        // ============================
        for (int i = 0; i < nodes.size(); ++i)
        {
            int p = nodes[i].parent;

            if (p >= 0)
            {
                long long key = edgeKey(i, p);

                if (visited.count(key) == 0)
                {
                    visited.insert(key);

                    roads.push_back({nodes[i].position,
                                     nodes[p].position});
                }
            }
        }

        // ============================
        // 2. extra links（闭环边）
        // ============================
        for (int i = 0; i < nodes.size(); ++i)
        {
            for (int j : nodes[i].extraLinks)
            {
                long long key = edgeKey(i, j);

                if (visited.count(key) == 0)
                {
                    visited.insert(key);

                    roads.push_back({nodes[i].position,
                                     nodes[j].position});
                }
            }
        }

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

    float SCANetwork::computeIdealAngleScore(int nodeID, int candidateID) const
    {
        const auto &node = nodes[nodeID];
        const auto &cand = nodes[candidateID];

        Eigen::Vector2f dir = (cand.position - node.position).normalized();

        std::vector<Eigen::Vector2f> existingDirs;

        // parent
        if (node.parent >= 0)
        {
            existingDirs.push_back(
                (nodes[node.parent].position - node.position).normalized());
        }

        // children
        for (int i = 0; i < nodes.size(); ++i)
        {
            if (nodes[i].parent == nodeID)
            {
                existingDirs.push_back(
                    (nodes[i].position - node.position).normalized());
            }
        }

        // extra links
        for (int j : node.extraLinks)
        {
            existingDirs.push_back(
                (nodes[j].position - node.position).normalized());
        }

        if (existingDirs.empty())
            return 1.0f;

        float minDot = 1.0f;

        for (auto &d : existingDirs)
        {
            float dot = std::abs(d.dot(dir));
            minDot = std::min(minDot, dot);
        }

        // 越小越好 → 转为分数
        return 1.0f - minDot;
    }

    bool SCANetwork::formsTriangle(int a, int b) const
    {
        std::unordered_set<int> na;

        // a 的邻居
        if (nodes[a].parent >= 0)
            na.insert(nodes[a].parent);

        for (int i = 0; i < nodes.size(); ++i)
            if (nodes[i].parent == a)
                na.insert(i);

        for (int j : nodes[a].extraLinks)
            na.insert(j);

        // b 的邻居
        std::unordered_set<int> nb;

        if (nodes[b].parent >= 0)
            nb.insert(nodes[b].parent);

        for (int i = 0; i < nodes.size(); ++i)
            if (nodes[i].parent == b)
                nb.insert(i);

        for (int j : nodes[b].extraLinks)
            nb.insert(j);

        // 是否有公共邻居 → 三角形
        for (int x : na)
            if (nb.count(x))
                return true;

        return false;
    }

    void SCANetwork::buildConnections()
    {
        buildKDTree();

        for (int i = 0; i < nodes.size(); ++i)
        {
            auto nearby = getNodesInRadius(nodes[i].position, connectDist);

            std::vector<std::pair<float, int>> candidates;

            for (int j : nearby)
            {
                if (j == i || j == nodes[i].parent)
                    continue;

                float dist = (nodes[j].position - nodes[i].position).norm();

                float angleScore = computeIdealAngleScore(i, j);

                float score = angleScore * idealAngleWeight + (1.0f / (dist + 1e-4f));

                candidates.emplace_back(score, j);
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](auto &a, auto &b)
                      { return a.first > b.first; });

            int added = 0;

            for (auto &c : candidates)
            {
                int j = c.second;

                if (formsTriangle(i, j))
                    continue;

                nodes[i].extraLinks.push_back(j);

                added++;
                if (added >= maxExtraLinks)
                    break;
            }
        }
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

}