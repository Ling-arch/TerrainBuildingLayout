// d:\Thesis\TerrainBuildingLayout\src\terrain\secondaryRoadGenerator.cpp
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

    void SCANetwork::update(bool& growthStopped)
    {
        growthStopped = false;
        // ============================
        // 参数
        // ============================
        const int MAX_NODES = 200000;
        const float STEP_SIZE = 2.0f;
        const float MIN_DIST2 = 4.0f;
        const float CONNECT_DIST = 6.0f;
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
            bool tooClose = false;

            for (const auto &n : nodes)
            {
                if ((n.position - newPos).squaredNorm() < MIN_DIST2)
                {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose)
            {
                for (const auto &n : newNodes)
                {
                    if ((n.position - newPos).squaredNorm() < MIN_DIST2)
                    {
                        tooClose = true;
                        break;
                    }
                }
            }

            if (tooClose)
                continue;

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
        for (int i = 0; i < newNodes.size(); ++i)
        {
            int newID = oldSize + i;
            auto &newNode = nodes[newID];

            auto nearby = getNodesInRadius(newNode.position, CONNECT_DIST);

            for (int nid : nearby)
            {
                // ❗不能连回自己父节点
                if (nid == newNode.parent)
                    continue;

                float d2 = (nodes[nid].position - newNode.position).squaredNorm();

                if (d2 < CONNECT_DIST2)
                {
                    // 🔥 关键：直接改 parent（真正连接）
                    newNode.parent = nid;

                    // 👉 可选：避免多次连接
                    break;
                }
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
        buildKDTree();
    }

    std::vector<std::vector<Eigen::Vector2f>> SCANetwork::extractRoads() const
    {
        if (nodes.empty())
            return {};
        std::vector<std::vector<Eigen::Vector2f>> roads;

        // ============================
        // 1. build children
        // ============================
        std::vector<std::vector<int>> children(nodes.size());

        for (int i = 0; i < nodes.size(); ++i)
        {
            int p = nodes[i].parent;
            if (p >= 0)
                children[p].push_back(i);
        }

        // ============================
        // 2. DFS
        // ============================
        std::function<void(int, std::vector<Eigen::Vector2f> &)> dfs;

        dfs = [&](int node, std::vector<Eigen::Vector2f> &path)
        {
            path.push_back(nodes[node].position);

            // ===== 叶子节点 =====
            if (children[node].empty())
            {
                if (path.size() > 1)
                    roads.push_back(path);
                return;
            }

            // ===== 分叉 =====
            for (int c : children[node])
            {
                std::vector<Eigen::Vector2f> newPath = path;
                dfs(c, newPath);
            }
        };

        // ============================
        // 3. 从 root 开始
        // ============================
        for (int i = 0; i < nodes.size(); ++i)
        {
            if (nodes[i].parent == -1) // root
            {
                std::vector<Eigen::Vector2f> path;
                dfs(i, path);
            }
        }

        return roads;
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
