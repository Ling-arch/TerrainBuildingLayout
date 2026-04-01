
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
        const auto &a = attractors[attractorID];

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

    std::vector<int> SCANetwork::getRelativeNeighbors(const Attractor &a) const
    {

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

        const float STEP_SIZE = 3.0f;
        const float CONNECT_DIST = 12.0f;
        const float CONNECT_DIST2 = CONNECT_DIST * CONNECT_DIST;

        const int MIN_BRANCH_GAP = 5; //  分叉间隔

        std::cout << "\n========== [SCA UPDATE BEGIN] ==========\n";

        // 1. Reset
        for (auto &a : attractors)
            a.resetFrame();

        for (auto &n : nodes)
            n.influencedBy.clear();

        // 2. Attractor → Node
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
        // 3. Grow Nodes（最终修正版）
        std::vector<SCANode> newNodes = growNodes(STEP_SIZE, MIN_BRANCH_GAP);
        int oldSize = nodes.size();

        // 插入新节点
        nodes.insert(nodes.end(), newNodes.begin(), newNodes.end());

        // 建立 link
        for (int i = 0; i < newNodes.size(); ++i)
        {
            int newID = oldSize + i;
            int parent = nodes[newID].parent;
            nodes[newID].links.push_back(parent);
            nodes[parent].links.push_back(newID);
        }

        // ============================
        // 5. Path 更新
        // ============================
        for (int i = 0; i < newNodes.size(); ++i)
        {
            int newID = oldSize + i;
            int parent = nodes[newID].parent;

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

        // // ============================
        // // 6. Loop（DEBUG增强版）
        // // ============================

        loopConnect(10, CONNECT_DIST, MIN_BRANCH_GAP);

        // std::cout << "\n========== [LOOP DEBUG END] ==========\n";
        // 7. Remove Attractors
        attractors.erase(
            std::remove_if(
                attractors.begin(),
                attractors.end(),
                [&](Attractor &a)
                {
                    return isAttractorKilled(a);
                }),
            attractors.end());

        std::cout << "========== [SCA UPDATE END] ==========\n";
    }

    std::vector<SCANode> SCANetwork::growNodes(float stepSize, int minBranchGap)
    {
        std::vector<SCANode> newNodes;
        std::vector<int> acceptedParents;
        auto getConstraintPathID = [&](int nodeID) -> int
        {
            const auto &n = nodes[nodeID];
            if (n.links.size() == 2)
            {
                if (!n.paths.empty())
                    return n.paths[0];
            }
            return -1;
        };
        // std::cout << "\n========== [GROW NODES DEBUG] ==========\n";
        for (int i = 0; i < nodes.size(); ++i)
        {
            auto &node = nodes[i];

            // std::cout << "\n--------------------------------------\n";
            // std::cout << "[Node] " << i
            //           << " links=" << node.links.size()
            //           << " paths=" << node.paths.size()
            //           << " influenced=" << node.influencedBy.size()
            //           << "\n";
            //  打印 links 具体
            // std::cout << "  links: ";
            // for (int l : node.links)
            // {
            //     std::cout << l << " ";
            // }
            // std::cout << "\n";
            //  打印 paths
            // std::cout << "  paths: ";
            // for (int p : node.paths)
            // {
            //     std::cout << p << " ";
            // }
            // std::cout << "\n";

            // 基础过滤
            if (node.influencedBy.empty())
                continue;

            if (node.links.size() >= 4)
                continue;

            Eigen::Vector2f dir = getAverageDirection(i);
            bool hasAttractor = !node.influencedBy.empty();

            // ============================
            // CASE 1: 正常方向
            // ============================
            if (dir.norm() >= 1e-6f)
            {
                dir.normalize();
            }

            // CASE 2: 有 attractor，但被抵消
            else if (hasAttractor)
            {
                // 👉 找最近 attractor
                float bestDist = FLT_MAX;
                int bestAid = -1;

                for (int aid : node.influencedBy)
                {
                    float d = (attractors[aid].position - node.position).squaredNorm();
                    if (d < bestDist)
                    {
                        bestDist = d;
                        bestAid = aid;
                    }
                }

                if (bestAid >= 0)
                {
                    dir = (attractors[bestAid].position - node.position).normalized();

                    std::cout << "  [Fallback: nearest attractor] Node[" << i << "] -> Attractor[" << bestAid << "]\n";
                }
            }

            // CASE 3: 没有 attractor → 惯性延伸
            else if (node.links.size() == 1)
            {
                dir = (node.position - nodes[node.links[0]].position).normalized();

                std::cout << "  [Fallback: inertia] Node[" << i << "]\n";
            }

            // ============================
            // CASE 4: 仍然失败 → 放弃
            if (dir.norm() < 1e-6f)
            {
                std::cout << "  [Skip: no valid direction] Node[" << i << "]\n";
                continue;
            }
            // std::cout << "  dir = (" << dir.x() << ", " << dir.y() << ")\n";
            bool isBranch = true;
            // ============================
            // 分叉约束（只针对 links==2）
            if (node.links.size() == 2)
            {
                // std::cout << "  [Branch Check] links == 2\n";
                int pid = getConstraintPathID(i);
                if (pid < 0)
                    continue;

                auto &path = paths[pid];

                auto it = std::find(path.begin(), path.end(), i);
                if (it == path.end())
                    continue;

                int curIdx = std::distance(path.begin(), it);
                // (A) 与已有分叉点
                int dist = findNearestBranchDist(pid, i);

                if (dist < minBranchGap)
                {
                    isBranch = false;
                }

                // }

                // (B) 与本轮新分叉
                if (isBranch)
                {
                    for (int prevParent : acceptedParents)
                    {
                        int prevPID = getConstraintPathID(prevParent);
                        if (prevPID != pid)
                            continue;
                        auto prevIt = std::find(path.begin(), path.end(), prevParent);
                        if (prevIt == path.end())
                            continue;
                        int prevIdx = std::distance(path.begin(), prevIt);
                        int gap = abs(prevIdx - curIdx);
                        // std::cout << "      vs Node[" << prevParent
                        //           << "] gap=" << gap << "\n";
                        if (gap < minBranchGap)
                        {
                            isBranch = false;
                            // std::cout << "      too close to NEW branch\n";
                            break;
                        }
                    }
                    // if (isBranch)
                    //     std::cout << "     pass new branch check\n";
                }

                if (isBranch)
                {
                    acceptedParents.push_back(i);
                }
            }

            // 最终是否生长
            if (!isBranch)
            {
                // std::cout << "   FINAL: skip grow (branch not allowed)\n";
                continue;
            }
            dir += Eigen::Vector2f(
                (rand() / (float)RAND_MAX - 0.5f) * 0.1f,
                (rand() / (float)RAND_MAX - 0.5f) * 0.1f);
            dir.normalize();
            // Grow
            Eigen::Vector2f newPos = node.position + getGuidedDirection(i, dir, 0.5f) * stepSize;
            SCANode newNode;
            newNode.position = newPos;
            newNode.parent = i;
            newNodes.push_back(newNode);
            node.isTip = false;
        }

        return newNodes;
    }

    bool SCANetwork::isAttractorKilled(const Attractor &a) const
    {
        auto neighbors = getRelativeNeighbors(a);
        auto killNodes = getNodesInRadius(a.position, killDist);

        if (neighbors.empty())
            return false; //  关键：不能删

        if (killNodes.empty())
            return false; //  你加的这个是对的

        for (int nid : neighbors)
        {
            if (std::find(killNodes.begin(), killNodes.end(), nid) == killNodes.end())
            {
                return false; // 有一个没覆盖 → 不删
            }
        }

        return true; // 全覆盖才删
    }

    void SCANetwork::loopConnect(int forbiddenGap, int CONNECT_DIST, int MIN_BRANCH_GAP)
    {
        buildKDTree();

        std::cout << "\n================ LOOP CONNECT BEGIN ================\n";

        std::unordered_set<int> validNodes = filterAllPathEndpoints(5);

        std::cout << "[Valid Endpoints] count = " << validNodes.size() << "\n  ";
        for (int n : validNodes)
            std::cout << n << " ";
        std::cout << "\n";

        std::unordered_set<int> connectedEndpoints;

        for (int curNodeID : validNodes)
        {
            if (connectedEndpoints.count(curNodeID))
                continue;

            auto &curNode = nodes[curNodeID];

            std::cout << "\n--------------------------------------------------\n";
            std::cout << "[PROCESS NODE] " << curNodeID
                      << " | links=" << curNode.links.size()
                      << " | path=" << curNode.paths[0] << "\n";

            if (curNode.links.size() > 1)
            {
                std::cout << "  -> SKIP: not endpoint\n";
                continue;
            }

            int parent = curNode.parent;
            std::vector<int> selfPath = paths[curNode.paths[0]];

            // =========================
            // 1. forbidden
            // =========================
            std::unordered_set<int> forbidden;

            for (int pid : nodes[parent].paths)
            {
                auto &path = paths[pid];

                auto it = std::find(path.begin(), path.end(), parent);
                int idx = std::distance(path.begin(), it);

                for (int k = -forbiddenGap; k <= forbiddenGap; ++k)
                {
                    int id = idx + k;
                    if (id >= 0 && id < path.size())
                    {
                        forbidden.insert(path[id]);
                    }
                }
            }

            // =========================
            // 一行打印
            // =========================
            std::cout << "  [Forbidden Nodes]: ";

            if (forbidden.empty())
            {
                std::cout << "none";
            }
            else
            {
                for (int nid : forbidden)
                    std::cout << nid << " ";
            }

            std::cout << "\n";

            // =========================
            // 2. nearby
            // =========================
            auto nearby = getNodesInRadius(curNode.position, CONNECT_DIST);

            std::vector<int> endpoints, branchNodes, midNodes;

            std::vector<int> validCandidates;

            // ============================
            // 1. 打印 Nearby（原始）
            // ============================
            std::cout << "  [Nearby] : ";
            for (int nid : nearby)
            {
                std::cout << nid << " ";
            }
            std::cout << "\n";

            // ============================
            // 2. 过滤 + 分类
            // ============================
            for (int nid : nearby)
            {
                if (nid == parent)
                    continue;
                if (forbidden.count(nid))
                    continue;
                if (std::find(selfPath.begin(), selfPath.end(), nid) != selfPath.end())
                    continue;

                validCandidates.push_back(nid);

                int deg = nodes[nid].links.size();

                if (deg == 1 && paths[nodes[nid].paths[0]].size() >= 5)
                {
                    endpoints.push_back(nid);
                }
                else if (deg == 3)
                {
                    branchNodes.push_back(nid);
                }
                else if (deg == 2)
                {
                    midNodes.push_back(nid);
                }
            }

            // ============================
            // 3. 打印 Candidates（总览）
            // ============================
            std::cout << "  [Candidates] : ";
            for (int nid : validCandidates)
            {
                std::cout << nid << "(d=" << nodes[nid].links.size() << ") ";
            }
            std::cout << "\n";

            // ============================
            // 4. 分类打印（关键！）
            // ============================
            auto printGroup = [&](const std::string &name, const std::vector<int> &arr)
            {
                std::cout << "    " << name << " : ";
                for (int nid : arr)
                    std::cout << nid << " ";
                std::cout << "\n";
            };

            printGroup("endpoints", endpoints);
            printGroup("branchNodes", branchNodes);
            printGroup("midNodes", midNodes);
            // =========================
            // 3. score
            // =========================
            auto score = [&](int nid)
            {
                Eigen::Vector2f dir1 = (curNode.position - nodes[parent].position).normalized();

                Eigen::Vector2f dir2 = (nodes[nid].position - curNode.position).normalized();

                return dir1.dot(dir2);
            };

            // =========================
            // 4. pick best
            // =========================
            int best = -1;
            float bestScore = -1;

            auto pickBest = [&](std::vector<int> &arr, const std::string &tag)
            {
                for (int nid : arr)
                {
                    float s = score(nid);

                    std::cout << "    [" << tag << "] "
                              << nid << " score=" << s << "\n";

                    if (s > bestScore)
                    {
                        bestScore = s;
                        best = nid;
                    }
                }
            };

            std::cout << "  [Selection]\n";

            if (!endpoints.empty())
            {
                std::cout << "    Priority: ENDPOINT\n";
                pickBest(endpoints, "endpoint");
            }
            else if (!branchNodes.empty())
            {
                std::cout << "    Priority: BRANCH\n";
                pickBest(branchNodes, "branch");
            }
            else
            {
                std::cout << "    Priority: MID\n";

                for (int nid : midNodes)
                {
                    int pid = nodes[nid].paths[0];
                    int dist = findNearestBranchDist(pid, nid);

                    std::cout << "    [mid] " << nid
                              << " branchDist=" << dist;

                    if (dist >= MIN_BRANCH_GAP)
                    {
                        float s = score(nid);
                        std::cout << " score=" << s << "\n";

                        if (s > bestScore)
                        {
                            bestScore = s;
                            best = nid;
                        }
                    }
                    else
                    {
                        std::cout << " -> rejected\n";
                    }
                }
            }

            if (best < 0)
            {
                std::cout << "  -> NO VALID TARGET\n";
                continue;
            }

            float dist = (nodes[best].position - curNode.position).norm();
            if (dist > CONNECT_DIST)
                continue;

            if (bestScore < 0)
                continue;

            std::cout << "  [SELECTED] " << best
                      << " score=" << bestScore << " dist=" << dist << "\n";

            std::cout << "  [LINK] " << curNodeID << " <-> " << best << "\n";

            // =========================
            // 6. path rebuild
            // =========================
            int curPID = curNode.paths[0];
            auto &curPath = paths[curPID];
            auto &targetNode = nodes[best];

            std::cout << "  [PATH UPDATE]\n";

            if (targetNode.links.size() == 1)
            {
                int pid = targetNode.paths[0];
                auto &path = paths[pid];

                std::cout << "    CASE: endpoint merge | pid=" << pid << "\n";

                if (path.front() == best)
                {
                    std::cout << "      direction: forward\n";
                    curPath.insert(curPath.end(), path.begin(), path.end());
                }
                else
                {
                    std::cout << "      direction: reverse\n";
                    curPath.insert(curPath.end(), path.rbegin(), path.rend());
                }

                for (int nid : path)
                    replacePathID(nid, pid, curPID);

                if (validNodes.count(best))
                    connectedEndpoints.insert(best);

                path.clear();
            }
            else if (targetNode.links.size() == 2)
            {
                std::cout << "    CASE: mid attach\n";

                // 找 curNode 在 path 中的位置
                auto it = std::find(curPath.begin(), curPath.end(), curNodeID);

                if (it == curPath.end())
                {
                    std::cout << "    [ERROR] curNode not in curPath!\n";
                    return;
                }

                bool isHead = (it == curPath.begin());
                bool isTail = (it == curPath.end() - 1);

                if (isTail)
                {
                    std::cout << "      attach at TAIL\n";
                    curPath.push_back(best);
                }
                else if (isHead)
                {
                    std::cout << "      attach at HEAD\n";
                    curPath.insert(curPath.begin(), best);
                }
                else
                {
                    std::cout << "      [ERROR] curNode in middle of path!\n";
                    return;
                }

                nodes[best].paths.push_back(curPID);
            }
            else if (targetNode.links.size() == 3)
            {
                std::cout << "    CASE: branch merge\n";

                // 找 curNode 在 curPath 的位置
                auto itCur = std::find(curPath.begin(), curPath.end(), curNodeID);

                if (itCur == curPath.end())
                {
                    std::cout << "    [ERROR] curNode not in curPath!\n";
                    return;
                }

                bool curIsHead = (itCur == curPath.begin());
                bool curIsTail = (itCur == curPath.end() - 1);

                if (!curIsHead && !curIsTail)
                {
                    std::cout << "    [ERROR] curNode in middle!\n";
                    return;
                }

                for (int pid : targetNode.paths)
                {
                    auto &path = paths[pid];

                    if (path.empty())
                        continue;

                    bool targetIsHead = (path.front() == best);
                    bool targetIsTail = (path.back() == best);

                    if (!targetIsHead && !targetIsTail)
                        continue;

                    int branchNear = -1;
                    if (targetIsHead && path.size() > 1)
                    {
                        branchNear = 1;
                    }
                    else if (targetIsTail && path.size() > 1)
                    {
                        branchNear = path.size() - 2;
                    }
                    if (branchNear < 0)
                        continue;
                    const SCANode &branchNbrNode = nodes[path[branchNear]];
                    float connectDot = (branchNbrNode.position - targetNode.position).dot(targetNode.position - curNode.position);
                    
                    if(connectDot < 0)
                        continue;
                    std::cout << "      merge pid=" << pid
                              << " | cur(" << (curIsHead ? "HEAD" : "TAIL") << ")"
                              << " target(" << (targetIsHead ? "HEAD" : "TAIL") << ")\n";

                    // =========================
                    // 4 cases
                    // =========================

                    if (curIsHead)
                    {
                        if (targetIsHead)
                        {
                            // case 1: head-head → reverse + front
                            curPath.insert(curPath.begin(), path.rbegin(), path.rend());
                            std::cout << "        action: reverse + insert FRONT\n";
                        }
                        else
                        {
                            // case 2: head-tail → forward + front
                            curPath.insert(curPath.begin(), path.begin(), path.end());
                            std::cout << "        action: forward + insert FRONT\n";
                        }
                    }
                    else // curIsTail
                    {
                        if (targetIsHead)
                        {
                            // case 3: tail-head → forward + back
                            curPath.insert(curPath.end(), path.begin(), path.end());
                            std::cout << "        action: forward + insert BACK\n";
                        }
                        else
                        {
                            // case 4: tail-tail → reverse + back
                            curPath.insert(curPath.end(), path.rbegin(), path.rend());
                            std::cout << "        action: reverse + insert BACK\n";
                        }
                    }

                    // =========================
                    // 更新 path ID
                    // =========================
                    for (int nid : path)
                    {
                        replacePathID(nid, pid, curPID);
                    }

                    path.clear();
                    break;
                }
            }

            //  link
            // =========================
            nodes[curNodeID].links.push_back(best);
            nodes[best].links.push_back(curNodeID);

            std::cout << "   DONE\n";
        }

        std::cout << "\n================ LOOP CONNECT END =================\n";
    }

    bool SCANetwork::passAngleConstraint(int nodeID, const Eigen::Vector2f &candidateDir, float threshold)
    {
        auto &node = nodes[nodeID];
        int deg = node.links.size();

        if (deg == 0)
            return true;

        // 收集 neighbor direction
        std::vector<Eigen::Vector2f> dirs;

        for (int nid : node.links)
        {
            Eigen::Vector2f d = (nodes[nid].position - node.position).normalized();
            dirs.push_back(d);
        }

        Eigen::Vector2f baseDir;

        // =========================
        // CASE 1
        if (deg == 1)
        {
            baseDir = dirs[0];
        }

        // =========================
        // CASE 2
        else if (deg == 2)
        {
            Eigen::Vector2f sum = dirs[0] + dirs[1];

            if (sum.norm() < 1e-3f)
            {
                // 对向 → 用垂直方向
                baseDir = Eigen::Vector2f(-dirs[0].y(), dirs[0].x());
            }
            else
            {
                baseDir = sum.normalized();
            }
        }

        // =========================
        // CASE 3
        else if (deg == 3)
        {
            Eigen::Vector2f sum(0, 0);
            for (auto &d : dirs)
                sum += d;

            baseDir = (-sum).normalized();
        }
        else
        {
            return false;
        }

        // =========================
        // 方向一致性（选正方向）
        if (baseDir.dot(candidateDir) < 0)
            baseDir = -baseDir;

        float dotVal = baseDir.dot(candidateDir);
        dotVal = std::clamp(dotVal, -1.0f, 1.0f);

        float angle = acos(dotVal);

        return angle < threshold * 0.5f;
    }

    void SCANetwork::replacePathID(int nodeID, int oldPID, int newPID)
    {
        auto &p = nodes[nodeID].paths;

        // 1. remove oldPID
        p.erase(std::remove(p.begin(), p.end(), oldPID), p.end());

        // 2. add newPID（避免重复）
        if (std::find(p.begin(), p.end(), newPID) == p.end())
        {
            p.push_back(newPID);
        }
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

        int minBranchDist = INT_MAX;

        for (int i = 0; i < path.size(); ++i)
        {
            int nid = path[i];

            if (nodes[nid].links.size() >= 3) // 分叉点
            {
                int d = abs(i - idx);
                if (d > 0)
                    minBranchDist = std::min(minBranchDist, d);
            }
        }

        return minBranchDist;
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

        for (const auto &path : paths)
        {
            if (path.empty())
                continue;
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

    Eigen::Vector2f SCANetwork::getGuidedDirection(
        int nodeID,
        const Eigen::Vector2f &candidateDir,
        float threshold)
    {
        auto &node = nodes[nodeID];
        int deg = node.links.size();

        // std::cout << "\n============================\n";
        // std::cout << "[GuidedDir] NodeID = " << nodeID << "\n";
        // std::cout << "deg = " << deg << "\n";

        // print links
        // std::cout << "links: ";
        // for (int nid : node.links)
        //     std::cout << nid << " ";
        // std::cout << "\n";

        // std::cout << "candidateDir = ("
        //           << candidateDir.x() << ", "
        //           << candidateDir.y() << ")\n";

        if (deg == 0)
        {
            // std::cout << "deg==0 -> return candidateDir\n";
            return candidateDir.normalized();
        }

        std::vector<Eigen::Vector2f> dirs;

        for (int nid : node.links)
        {
            Eigen::Vector2f d = (nodes[nid].position - node.position).normalized();
            dirs.push_back(d);
        }

        Eigen::Vector2f baseDir;

        // =========================
        // CASE 1
        // =========================
        if (deg == 1)
        {
            baseDir = -dirs[0];

            float blend = std::clamp(threshold + 0.2f, 0.55f, 0.75f);

            // std::cout << "[CASE 1] endpoint\n";
            // std::cout << "raw baseDir = (" << baseDir.x() << ", " << baseDir.y() << ")\n";
            // std::cout << "blend = " << blend << "\n";

            baseDir = (baseDir * blend + candidateDir * (1 - blend)).normalized();

            // std::cout << "blended baseDir = (" << baseDir.x() << ", " << baseDir.y() << ")\n";
        }

        // =========================
        // CASE 2
        // =========================
        // else if (deg == 2 || deg == 3)
        // {
        //     Eigen::Vector2f sum = dirs[0] + dirs[1];

        //     std::cout << "[CASE 2] potential branch\n";
        //     std::cout << "dir0 = (" << dirs[0].x() << "," << dirs[0].y() << ")\n";
        //     std::cout << "dir1 = (" << dirs[1].x() << "," << dirs[1].y() << ")\n";

        //     if (sum.norm() < 1e-3f)
        //     {
        //         baseDir = Eigen::Vector2f(-dirs[0].y(), dirs[0].x());
        //         std::cout << "opposite case -> perpendicular baseDir\n";
        //     }
        //     else
        //     {
        //         baseDir = sum.normalized();
        //         std::cout << "sum baseDir\n";
        //     }

        //     std::cout << "raw baseDir = (" << baseDir.x() << "," << baseDir.y() << ")\n";
        //     std::cout << "threshold blend = " << threshold << "\n";
        //     float dot = baseDir.dot(candidateDir);
        //     if (dot < 0)
        //         baseDir = -baseDir;
        //     baseDir = (baseDir * threshold + candidateDir * (1 - threshold)).normalized();

        //     std::cout << "blended baseDir = (" << baseDir.x() << "," << baseDir.y() << ")\n";
        // }

        // =========================
        // CASE 3
        // =========================
        else if (deg == 2 || deg == 3)
        {
            Eigen::Vector2f sum(0, 0);

            // std::cout << "[CASE 3] branch junction\n";

            for (auto &d : dirs)
            {
                sum += d;
                // std::cout << "dir = (" << d.x() << "," << d.y() << ")\n";
            }
            if (sum.norm() < 1e-3f)
            {
                baseDir = Eigen::Vector2f(-dirs[0].y(), dirs[0].x());
                // std::cout << "opposite case -> perpendicular baseDir\n";
            }
            else
            {
                baseDir = sum.normalized();
                // std::cout << "sum baseDir\n";
            }
            // std::cout << "raw baseDir (missing dir) = (" << baseDir.x() << "," << baseDir.y() << ")\n";
            // std::cout << "threshold blend = " << threshold << "\n";
            float dot = baseDir.dot(candidateDir);
            if (dot < 0)
                baseDir = -baseDir;
            baseDir = (baseDir * threshold + candidateDir * (1 - threshold)).normalized();

            // std::cout << "blended baseDir = (" << baseDir.x() << "," << baseDir.y() << ")\n";
        }
        else
        {
            // std::cout << "[CASE DEFAULT] return candidateDir\n";
            return candidateDir.normalized();
        }

        // std::cout << "FINAL guidedDir = (" << baseDir.x() << "," << baseDir.y() << ")\n";
        // std::cout << "============================\n";

        return baseDir.normalized();
    }

    std::unordered_set<int> SCANetwork::filterAllPathEndpoints(int minLen) const
    {
        std::unordered_set<int> result;
        std::cout << "\n================ FILTER PATH ENDPOINTS ================\n";
        for (int pid = 0; pid < paths.size(); ++pid)
        {
            const auto &path = paths[pid];

            if (path.empty())
            {
                // std::cout << "\n[Path " << pid << "] EMPTY -> skip\n";
                continue;
            }

            // std::cout << "\n--------------------------------------------------\n";
            std::cout << "[Path " << pid << "] size=" << path.size() << "\n";

            // =========================
            // 打印 path
            // =========================
            std::cout << "  nodes: ";
            for (int nid : path)
                std::cout << nid << " ";
            std::cout << "\n";

            // =========================
            // 找 branch nodes
            // =========================
            std::vector<int> branchIdx;

            for (int i = 0; i < (int)path.size(); ++i)
            {
                int nid = path[i];
                int deg = nodes[nid].links.size();

                if (deg >= 3)
                    branchIdx.push_back(i);
            }

            // std::cout << "  branchIdx: ";
            // for (int idx : branchIdx)
            //     std::cout << idx << "(node " << path[idx] << ") ";
            // std::cout << "\n";

            // =========================
            // CASE 1: 没有分叉
            // =========================
            if (branchIdx.empty())
            {
                int len = path.size();
                // std::cout << "  [No Branch] len=" << len << "\n";
                if (len >= minLen)
                {
                    int head = path.front();
                    int tail = path.back();

                    // std::cout << "    head=" << head
                    //           << " deg=" << nodes[head].links.size() << "\n";

                    // std::cout << "    tail=" << tail
                    //           << " deg=" << nodes[tail].links.size() << "\n";

                    if (nodes[head].links.size() == 1)
                    {
                        result.insert(head);
                        // std::cout << "    -> ADD head\n";
                    }

                    if (nodes[tail].links.size() == 1)
                    {
                        result.insert(tail);
                        // std::cout << "    -> ADD tail\n";
                    }
                }
                // else
                // {
                //     std::cout << "    -> skip (too short)\n";
                // }

                continue;
            }

            // =========================
            // CASE 2: 从头往里找 branch
            // =========================
            int firstBranchIdx = branchIdx.front();

            int lenHead = firstBranchIdx + 1; //  修复长度

            // std::cout << "  [Head Segment] endIdx=" << firstBranchIdx
            //           << " len=" << lenHead << "\n";

            int headNode = path.front();

            if (lenHead >= minLen)
            {
                // std::cout << "    head=" << headNode
                //           << " deg=" << nodes[headNode].links.size() << "\n";

                if (nodes[headNode].links.size() == 1)
                {
                    result.insert(headNode);
                    // std::cout << "    -> ADD head\n";
                }
                // else
                // {
                //     std::cout << "    -> skip (not endpoint)\n";
                // }
            }
            // else
            // {
            //     std::cout << "    -> skip (too short)\n";
            // }

            // =========================
            // CASE 3: 从尾往里找 branch
            // =========================
            int lastBranchIdx = branchIdx.back();

            int lenTail = path.size() - lastBranchIdx; //  修复长度

            // std::cout << "  [Tail Segment] startIdx=" << lastBranchIdx
            //           << " len=" << lenTail << "\n";

            int tailNode = path.back();

            if (lenTail >= minLen)
            {
                // std::cout << "    tail=" << tailNode
                //           << " deg=" << nodes[tailNode].links.size() << "\n";

                if (nodes[tailNode].links.size() == 1)
                {
                    result.insert(tailNode);
                    // std::cout << "    -> ADD tail\n";
                }
                else
                {
                    // std::cout << "    -> skip (not endpoint)\n";
                }
            }
            // else
            // {
            //     // std::cout << "    -> skip (too short)\n";
            // }
        }

        // std::cout << "\n[RESULT endpoints] count=" << result.size() << "\n  ";
        // for (int n : result)
        //     std::cout << n << " ";
        // std::cout << "\n";

        // std::cout << "=====================================================\n";

        return result;
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