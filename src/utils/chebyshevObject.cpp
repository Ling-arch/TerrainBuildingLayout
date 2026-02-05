#include "chebyshevObject.h"

namespace infinityVoronoi
{

    //                 2---n---3
    //                 | \   / |
    //                 w   0   e
    //                 | /   \ |
    //                 1---s---4
    // dirs = {0,1,2,3} = {e,n,w,s};
    // di = 0---e(左侧的三角形)

    constexpr int VERT_IDX[4][3] = {
        {0, 3, 4},
        {0, 2, 3},
        {0, 1, 2},
        {0, 4, 1}};

    constexpr int EDGE_CENTER_IDX[4][3] = {
        {6, 0, 7},
        {5, 1, 6},
        {4, 2, 5},
        {7, 3, 4}};

    constexpr int EDGE_NORMAL_IDX[4][3] = {
        {5, 0, 4},
        {4, 1, 7},
        {7, 2, 6},
        {6, 3, 5}};

    const std::vector<Vec2> INIT_CELL_VERTS = {{0, 0}, {-1, -1}, {-1, 1}, {1, 1}, {1, -1}};
    const std::vector<Vec2> EDGE_CENTERS = {Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0), Vec2(0, -1),
                                            INIT_CELL_VERTS[1] * 0.5f,
                                            INIT_CELL_VERTS[2] * 0.5f,
                                            INIT_CELL_VERTS[3] * 0.5f,
                                            INIT_CELL_VERTS[4] * 0.5f};

    std::vector<Vec2> EDGE_NORMALS = {Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0), Vec2(0, -1), INIT_CELL_VERTS[1], INIT_CELL_VERTS[2], INIT_CELL_VERTS[3], INIT_CELL_VERTS[4]};

    bool sectorsDoIntersect(const std::vector<MatrixXf> &sGeometry)
    {
        if (sGeometry.size() == 2)
        {
            return geo::trianglesDoIntersect2D(
                sGeometry[0],
                sGeometry[1]);
        }
        else if (sGeometry.size() == 4)
        {
            // 以后 pyrasDoIntersect
            return false;
        }

        return false;
    }

    void clipCellGeometry(
        SectorCutObject &cellSec,
        const Vec &siteA,
        const Vec &MvecA,
        float lambdaA,
        const std::vector<Vec> &sitesB,
        const std::vector<Eigen::MatrixXf> &MvecsB,
        const std::vector<Vec> &lambdasB,
        float domainExtent)
    {
        // 每个 polygon 的 centroid
        std::vector<Vec> polysCentroids =
            cellSec.getPolysCentroids(true);

        int polyCount = (int)polysCentroids.size();

        std::vector<float> distsA(polyCount);
        std::vector<float> distsL2A(polyCount);
        std::vector<bool> msk(polyCount, true);

        // ---------- 距离 siteA ----------
        for (int i = 0; i < polyCount; ++i)
        {
            Vec diff = polysCentroids[i] - siteA;

            distsA[i] = diff.dot(MvecA) / lambdaA;
            distsL2A[i] = diff.norm();

            // domain clipping
            if (polysCentroids[i].cwiseAbs().maxCoeff() >= domainExtent)
                msk[i] = false;
        }

        // ---------- neighbor sites ----------
        int neighborCount = (int)sitesB.size();

        for (int nb = 0; nb < neighborCount; ++nb)
        {
            const Vec &siteB = sitesB[nb];
            const auto &MvecB = MvecsB[nb];
            const Vec &lambdaB = lambdasB[nb];

            for (int i = 0; i < polyCount; ++i)
            {
                Vec diff = polysCentroids[i] - siteB;

                float maxDistB = -std::numeric_limits<float>::max();

                // max(dot(diff, MvecB_row)/lambda)
                for (int k = 0; k < MvecB.rows(); ++k)
                {
                    float d =
                        diff.dot(MvecB.row(k)) / lambdaB[k];

                    maxDistB = std::max(maxDistB, d);
                }

                bool dMsk = distsA[i] < maxDistB;

                // equality fallback (L2)
                if (std::fabs(distsA[i] - maxDistB) < 1e-6)
                {
                    float distsL2B = diff.norm();
                    if (distsL2A[i] < distsL2B)
                        dMsk = true;
                }

                msk[i] = msk[i] && dMsk;
            }
        }

        cellSec.setPolyIoLabels(msk);
    }

    DissolveResult dissolve(
        const std::vector<std::unique_ptr<SectorCutObject>> &sectors,
        int nDim)
    {
        DissolveResult result;

        if (nDim == 2 && sectors.size() == 4)
        {
            std::vector<std::vector<Vec>> parts;

            int sIdx = 0;
            for (auto &sec : sectors)
            {
                if (!sec)
                {
                    ++sIdx;
                    continue;
                }

                const auto &hv = sec->getHullVerts();
                size_t totalVerts = 0;
                for (size_t i = 0; i < hv.size(); ++i)
                {
                    totalVerts += hv[i].size();
                }
                parts.insert(parts.end(), hv.begin(), hv.end());

                auto keys = sec->hullPlaneKeys;

                result.planeKeys.insert(
                    result.planeKeys.end(),
                    keys.begin(), keys.end());

                ++sIdx;
            }

            if (parts.empty())
            {
                std::cout << " [dissolve]  WARNING: no parts collected\n";
                return result;
            }

            auto merged = chebyshevUtils::concatPolyParts(parts);

            // std::cout << " [dissolve]  merged polys = "
            //           << merged.size()
            //           << std::endl;

            for (size_t i = 0; i < merged.size(); ++i)
            {

                auto poly = chebyshevUtils::limitedDissolve2D(merged[i]);

                // std::cout << " [dissolve]  after dissolve verts = "
                //           << poly.size()
                //           << std::endl;

                if (!poly.empty())
                    result.polys.push_back(std::move(poly));
                else
                    std::cout
                        << "[dissolve]  WARNING: poly vanished after dissolve\n";
            }

            // std::cout << " [dissolve]  final poly count = "
            //           << result.polys.size()
            //           << std::endl;

            return result;
        }

        std::cout << " [dissolve]  unsupported configuration\n";
        return result;
    }

    TriCutObject::TriCutObject(
        const Vec2 &site,
        int di,
        const std::vector<float> &scale,
        const Eigen::Matrix2f &M)
    {
        initCellVerts = INIT_CELL_VERTS;

        // =========================================================
        // edges / polys / edgePolyIdxs
        // =========================================================
        edges = {{0, 1}, {1, 2}, {2, 0}};

        polys.clear();
        polys[1] = {0, 1, 2};

        edgePolyIdxs.clear();
        edgePolyIdxs = {
            {1, -1},
            {1, -1},
            {1, -1}};

        // =========================================================
        // +x,+y,-x,-y四个方向的scale
        // =========================================================
        std::vector<Vec2> vertScales(3);
        std::vector<Vec2> eCenterScales(3);
        std::vector<Vec2> eNormalScales(3);

        if (di % 2)
        {
            vertScales = {
                {1, 1},
                {scale[(di + 1) % 4], scale[di]},
                {scale[(di + 3) % 4], scale[di]}};
            eCenterScales = {
                {scale[(di + 1) % 4], scale[di]},
                {1, scale[di]},
                {scale[(di + 3) % 4], scale[di]}};
            eNormalScales = {
                {scale[di], scale[(di + 1) % 4]},
                {1, 1},
                {scale[di], scale[(di + 3) % 4]}};
        }
        else
        {
            vertScales = {
                {1, 1},
                {scale[di], scale[(di + 1) % 4]},
                {scale[di], scale[(di + 3) % 4]}};
            eCenterScales = {
                {scale[di], scale[(di + 1) % 4]},
                {scale[di], 1},
                {scale[di], scale[(di + 3) % 4]}};
            eNormalScales = {
                {scale[(di + 1) % 4], scale[di]},
                {1, 1},
                {scale[(di + 3) % 4], scale[di]}};
        }

        vertices.resize(3);
        std::vector<Vec2> edgeCenters(3);
        std::vector<Vec2> edgeNormals(3);
        edgesPlanes.clear();
        edgePlaneKeys = {-4, -5, -6};
        for (int i = 0; i < 3; ++i)
        {
            int vidx = VERT_IDX[di][i];
            Vec2 v = initCellVerts[vidx].cwiseProduct(vertScales[i]);
            vertices[i] = site + M.transpose() * v;
            edgeCenters[i] = site + M.transpose() * (EDGE_CENTERS[EDGE_CENTER_IDX[di][i]].cwiseProduct(eCenterScales[i]));
            edgeNormals[i] = (site + M.transpose() * (EDGE_NORMALS[EDGE_NORMAL_IDX[di][i]].cwiseProduct(eNormalScales[i]))).normalized();
            int key = -(i + 4);
            edgesPlanes[key] = {edgeCenters[i], edgeNormals[i]};
        }
    }

    void TriCutObject::clipWithPlane(
        const Vec2 &o,
        const Vec2 &n,
        int cutPlaneKey)
    {
        cutWithPlane(o, n, cutPlaneKey);
    }

    void TriCutObject::cutWithPlane(
        const Vec2 &o,
        const Vec2 &n,
        int cutPlaneKey)
    {
        const float eps = 1e-6f;
        DBG("  [TriCutObject::cutWithPlane]");
        DBG("    cutPlaneKey = " << cutPlaneKey);
        DBG("    o = [" << o.transpose() << "]");
        DBG("    n = [" << n.transpose() << "]");
        // ---- vertex signs ----
        Eigen::VectorXf dots(vertices.size());
        for (int i = 0; i < vertices.size(); ++i)
            dots[i] = (vertices[i] - o).dot(n);

        Eigen::VectorXi vMasks = chebyshevUtils::simpleSign(dots, eps);

        if ((vMasks.array() > 0).all() || (vMasks.array() < 0).all())
            return;

        // ---- edge masks ----
        std::vector<Vec2i> edgeMasks(edges.size());
        for (int i = 0; i < edges.size(); ++i)
            edgeMasks[i] = {vMasks[edges[i].x()], vMasks[edges[i].y()]};

        auto edgeHashs = chebyshevUtils::cantorPiV(edges);

        // ---- split polys ----
        std::map<int, std::vector<int>> newPolys;
        std::unordered_set<int> cutPolyKeys;

        for (auto &[pk, eIdxs] : polys)
        {
            std::unordered_set<int> signs;
            for (int eIdx : eIdxs)
            {
                signs.insert(edgeMasks[eIdx][0]);
                signs.insert(edgeMasks[eIdx][1]);
            }

            if (signs.count(1) && signs.count(-1))
            {
                cutPolyKeys.insert(pk);
                newPolys[pk * 2] = {};
                newPolys[pk * 2 + 1] = {};
            }
            else
            {
                newPolys[pk] = eIdxs;
            }
        }

        // ---- classify edges ----
        std::unordered_map<uint64_t, Vec2i> oldEdgesMasks;
        std::unordered_map<uint64_t, Vec2i> cutEdgesMasks;
        std::vector<Vec2i> newEdgePolyIdxs;

        for (int i = 0; i < edgeMasks.size(); ++i)
        {
            const Vec2i &em = edgeMasks[i];
            const Vec2i &epi = edgePolyIdxs[i];
            uint64_t h = edgeHashs[i];

            // 分支1：all(edgeMask <= 0) （负分支）
            if ((em[0] <= 0) && (em[1] <= 0))
            {
                oldEdgesMasks[h] = em;
                Vec2i newEpi = epi;

                // 判断epi[0]和epi[1]的切割状态
                bool epi0Cut = cutPolyKeys.count(epi[0]);
                bool epi1Cut = cutPolyKeys.count(epi[1]);

                if (epi0Cut && epi1Cut)
                {
                    // epi * 2 → 每个元素×2
                    newEpi = epi * 2;
                }
                else if (epi0Cut)
                {
                    // epi * [2, 1] → [epi[0]*2, epi[1]*1]
                    newEpi[0] = epi[0] * 2;
                    newEpi[1] = epi[1] * 1;
                }
                else if (epi1Cut)
                {
                    // Python: epi * [1, 2] → [epi[0]*1, epi[1]*2]
                    newEpi[0] = epi[0] * 1;
                    newEpi[1] = epi[1] * 2;
                }
                // 都不切割则保留原epi

                newEdgePolyIdxs.push_back(newEpi);
            }
            // 分支2：all(edgeMask >= 0) （正分支）
            else if ((em[0] >= 0) && (em[1] >= 0))
            {
                oldEdgesMasks[h] = em;
                Vec2i newEpi = epi;

                bool epi0Cut = cutPolyKeys.count(epi[0]);
                bool epi1Cut = cutPolyKeys.count(epi[1]);

                if (epi0Cut && epi1Cut)
                {
                    // epi * 2 + 1 → 每个元素×2+1
                    newEpi = epi * 2 + Vec2i(1, 1);
                }
                else if (epi0Cut)
                {
                    // epi * [2, 1] + [1, 0] → [epi0*2+1, epi1*1+0]
                    newEpi[0] = epi[0] * 2 + 1;
                    newEpi[1] = epi[1] * 1 + 0;
                }
                else if (epi1Cut)
                {
                    // epi * [1, 2] + [0, 1] → [epi0*1+0, epi1*2+1]
                    newEpi[0] = epi[0] * 1 + 0;
                    newEpi[1] = epi[1] * 2 + 1;
                }
                // 都不切割则保留原epi

                newEdgePolyIdxs.push_back(newEpi);
            }
            // 分支3：既有正又有负 → 切割边
            else
            {
                cutEdgesMasks[h] = em;
                newEdgePolyIdxs.push_back(epi);
            }
        }

        // ---- split edges ----
        int numVerts = vertices.size();
        std::vector<int> cutPlaneKeys;
        std::unordered_map<uint64_t, std::pair<int, int>> edgesReplaced;
        std::vector<std::pair<int, int>> edgeUpdates;

        for (int pk : cutPolyKeys)
        {
            std::vector<int> newEdgeInner;

            for (int eIdx : polys[pk])
            {
                auto &e = edges[eIdx];
                uint64_t h = edgeHashs[eIdx];

                for (int k = 0; k < 2; ++k)
                    if (vMasks[e[k]] == 0 &&
                        std::find(newEdgeInner.begin(), newEdgeInner.end(), e[k]) == newEdgeInner.end())
                        newEdgeInner.push_back(e[k]);

                if (cutEdgesMasks.count(h))
                {
                    int newVertIdx, newEdgeIdx;

                    if (edgesReplaced.count(h))
                    {
                        auto p = edgesReplaced[h];
                        newVertIdx = p.first;
                        newEdgeIdx = p.second;
                    }
                    else
                    {
                        newVertIdx = numVerts++;
                        newEdgeIdx = edges.size();

                        edgesReplaced[h] = {newVertIdx, newEdgeIdx};
                        cutPlaneKeys.push_back(edgePlaneKeys[eIdx]);

                        edgeUpdates.emplace_back(eIdx, newVertIdx);
                        edges.push_back({newVertIdx, e[1]});
                        edgePlaneKeys.push_back(edgePlaneKeys[eIdx]);

                        // edgePolyIdxs 映射
                        auto mask = cutEdgesMasks[h];
                        Vec2i old = newEdgePolyIdxs[eIdx];
                        Vec2i first, second;

                        if (mask[0] > 0 && mask[1] < 0)
                        {
                            // append(epi * 2)
                            first = old * 2;
                            // epi *= 2; epi += 1
                            second = old * 2 + Vec2i::Ones();
                        }
                        else
                        {
                            // append(epi * 2 + 1)
                            first = old * 2 + Vec2i::Ones();
                            // epi *= 2
                            second = old * 2;
                        }

                        newEdgePolyIdxs.push_back(first);
                        newEdgePolyIdxs[eIdx] = second;
                    }

                    newEdgeInner.push_back(newVertIdx);

                    // split edge 加回 polys
                    const Vec2i &epi2 = newEdgePolyIdxs[eIdx];
                    for (int nepi : newEdgePolyIdxs[eIdx])
                    {
                        // int nepi = epi2[k];
                        if (nepi > 0)
                        {
                            // 1. 获取该多边形对应的边列表（不存在则自动初始化空列表）
                            std::vector<int> &edgeList = newPolys[nepi];
                            // 2. 检查eIdx是否已存在于列表中
                            if (std::find(edgeList.begin(), edgeList.end(), eIdx) == edgeList.end())
                            {
                                edgeList.push_back(eIdx); // 不存在则添加
                            }
                        }
                    }
                    const Vec2i &epi2_new = newEdgePolyIdxs[newEdgeIdx];
                    for (int nepi : newEdgePolyIdxs[newEdgeIdx])
                    {
                        // int nepi = epi2_new[k];

                        if (nepi > 0)
                        {
                            // 1. 获取该多边形对应的边列表（不存在则自动初始化空列表）
                            std::vector<int> &edgeList = newPolys[nepi];
                            // 2. 检查eIdx是否已存在于列表中
                            if (std::find(edgeList.begin(), edgeList.end(), eIdx) == edgeList.end())
                            {
                                edgeList.push_back(newEdgeIdx); // 不存在则添加
                            }
                        }
                    }
                }
                else
                {
                    auto em = oldEdgesMasks[h];
                    if ((em.array() <= 0).all())
                        newPolys[pk * 2].push_back(eIdx);
                    else if ((em.array() >= 0).all())
                        newPolys[pk * 2 + 1].push_back(eIdx);
                }
            }
            newPolys[pk * 2].push_back(edges.size());
            newPolys[pk * 2 + 1].push_back(edges.size());
            newEdgePolyIdxs.push_back({pk * 2, pk * 2 + 1});
            // new cut edge
            edges.push_back({newEdgeInner[0], newEdgeInner[1]});
            edgePlaneKeys.push_back(cutPlaneKey);
        }

        edgesPlanes[cutPlaneKey] = {o, n};

        for (auto &[eIdx, vIdx] : edgeUpdates)
            edges[eIdx][1] = vIdx;

        if (!cutPlaneKeys.empty())
        {
            std::vector<std::pair<Vec2, Vec2>> planes;
            // std::cout << "cut plane keys are ";
            for (int k : cutPlaneKeys)
            {
                planes.push_back({edgesPlanes[k].o, edgesPlanes[k].n});
            }
            auto newVerts = chebyshevUtils::intersectLinesLine2D(planes, o, n);
            vertices.insert(vertices.end(), newVerts.begin(), newVerts.end());
        }

        polys = newPolys;
        edgePolyIdxs = newEdgePolyIdxs;
        DBG("    ---- polys (after cut) ----");
        // for (auto &[pk, eIdxs] : polys)
        // {
        //     std::ostringstream oss;
        //     oss << "      poly[" << pk << "] edges = ";
        //     for (int e : eIdxs)
        //         oss << e << " ";
        //     DBG(oss.str());
        // }

        DBG("    ---- polys (after cut) ----");

        for (auto &[pk, eIdxs] : polys)
        {
            DBG("      poly[" << pk << "]");

            for (int eIdx : eIdxs)
            {
                if (eIdx < 0 || eIdx >= edges.size())
                {
                    DBG("        edge[" << eIdx << "] <invalid>");
                    continue;
                }

                const Vec2i &e = edges[eIdx];
                int v0 = e[0];
                int v1 = e[1];

                DBG("        edge[" << eIdx << "] : "
                                    << v0 << " -> " << v1);

                if (v0 >= 0 && v0 < vertices.size())
                {
                    DBG("          v" << v0 << " = "
                                      << vec2ToStr(vertices[v0]));
                }
                else
                {
                    DBG("          v" << v0 << " = <invalid>");
                }

                if (v1 >= 0 && v1 < vertices.size())
                {
                    DBG("          v" << v1 << " = "
                                      << vec2ToStr(vertices[v1]));
                }
                else
                {
                    DBG("          v" << v1 << " = <invalid>");
                }
            }
        }
    }

    void TriCutObject::computePolysCentroidsAndWeights()
    {
        polysCentroids.clear();
        polysAreas.clear();

        for (const auto &[pk, eIdxs] : polys)
        {
            std::vector<std::array<int, 2>> es;
            es.reserve(eIdxs.size());
            for (int ei : eIdxs)
                es.push_back({edges[ei].x(), edges[ei].y()});

            auto pathOpt = chebyshevUtils::edgesToPath(es);

            if (!pathOpt.has_value())
            {
                continue;
            }

            const auto &path = pathOpt->path;

            std::vector<Vec2> poly;
            poly.reserve(path.size());
            for (int vid : path)
                poly.push_back(vertices[vid]);

            auto [c, a] = chebyshevUtils::computePolygonCentroid2D(poly, true);

            polysCentroids.push_back(c);
            polysAreas.push_back(a);
        }
    }

    std::vector<Vec> TriCutObject::getPolysCentroids(bool ioClipped)
    {
        if (polysCentroids.empty())
            computePolysCentroidsAndWeights();

        std::vector<Vec> res;

        if (ioClipped && !cellPolyIdxs.empty())
        {
            res.reserve(cellPolyIdxs.size());

            for (int idx : cellPolyIdxs)
            {
                Vec v(2);
                v = polysCentroids[idx];
                res.push_back(v);
            }
        }
        else
        {
            res.reserve(polysCentroids.size());

            for (const auto &p : polysCentroids)
            {
                Vec v(2);
                v = p;
                res.push_back(v);
            }
        }

        return res;
    }

    std::vector<float> TriCutObject::getPolysWeights(bool ioClipped)
    {
        if (polysAreas.empty())
            computePolysCentroidsAndWeights();

        if (ioClipped && !cellPolyIdxs.empty())
        {
            std::vector<float> res;
            res.reserve(cellPolyIdxs.size());
            for (int idx : cellPolyIdxs)
                res.push_back(polysAreas[idx]);
            return res;
        }

        return polysAreas;
    }

    std::vector<std::vector<Vec>> TriCutObject::getHullVerts()
    {
        std::unordered_map<int, std::vector<Vec2i>> es;

        // 1. collect hull edges
        for (size_t i = 0; i < edges.size(); ++i)
        {
            Vec2i epi = edgePolyIdxs[i];
            if ((epi.array() != 0).count() == 1)
            {
                int pk = edgePlaneKeys[i];
                es[pk].push_back(edges[i]);
            }
        }

        // init state
        if (es.empty())
        {
            hullPlaneKeys = {-6};
            return {{vertices[edges.back()[0]],
                     vertices[edges.back()[0]]}};
        }

        // 2. collect hull segments
        std::vector<Vec2i> segs;
        for (auto &[pk, eList] : es)
        {
            auto segments = chebyshevUtils::findConnectedEdgeSegments(eList);
            for (auto &seg : segments)
            {
                auto pathOpt = chebyshevUtils::edgesToPath(chebyshevUtils::convertVector2array(seg));
                if (!pathOpt)
                    continue;
                auto &path = pathOpt->path;
                segs.push_back({path.front(), path.back()});
            }
        }

        hullPlaneKeys.clear();
        for (auto &[pk, _] : es)
            hullPlaneKeys.push_back(pk);

        // 3. segments → closed paths
        auto paths = chebyshevUtils::edgesToPaths(segs);

        std::vector<std::vector<Vec>> res;
        res.reserve(paths.size());

        for (auto &p : paths)
        {
            std::vector<Vec> poly;
            poly.reserve(p.size());

            for (int vid : p)
            {
                Vec v(2);
                v << vertices[vid].x(), vertices[vid].y();
                poly.push_back(v);
            }

            res.push_back(std::move(poly));
        }

        return res;
    }

    std::vector<Vec> TriCutObject::getVertices()
    {
        std::vector<Vec> verts;
        verts.reserve(vertices.size());
        for (const auto &v : vertices)
        {
            Vec v_new(2);
            v_new << v.x(), v.y();
            verts.push_back(v_new);
        }
        return verts;
    }

    void TriCutObject::setPolyIoLabels(const std::vector<bool> &msk)
    {
        if (polysIoLabel.empty())
        {
            for (auto &[pk, _] : polys)
                polysIoLabel[pk] = true;
        }

        cellPolyIdxs.clear();

        int pIdx = 0;
        for (auto &[pk, _] : polys)
        {
            bool io = msk[pIdx];
            polysIoLabel[pk] = io;

            if (io)
            {
                cellPolyIdxs.push_back(pIdx);
            }
            else
            {
                for (auto &epi : edgePolyIdxs)
                {
                    if (epi[0] == pk)
                        epi[0] = 0;
                    if (epi[1] == pk)
                        epi[1] = 0;
                }
            }
            ++pIdx;
        }
    }

    std::vector<Vec2> TriCutObject::buildPolyFromKey(int key) const
    {
        std::vector<Vec2> poly;

        // ---------- 1. 取出该 poly 的 edge 索引 ----------
        auto it = polys.find(key);
        if (it == polys.end())
            return poly;

        const std::vector<int> &edgeIdxs = it->second;
        if (edgeIdxs.empty())
            return poly;

        // ---------- 2. 建立 vertex -> edges 邻接 ----------
        std::unordered_map<int, std::vector<int>> v2edges;
        v2edges.reserve(edgeIdxs.size() * 2);

        for (int eIdx : edgeIdxs)
        {
            const auto &e = edges[eIdx];
            v2edges[e[0]].push_back(eIdx);
            v2edges[e[1]].push_back(eIdx);
        }

        // ---------- 3. 选择起始 edge 和起始 vertex ----------
        int startEdge = edgeIdxs[0];
        const auto &e0 = edges[startEdge];

        int startV = e0[0];
        int currV = startV;
        int prevEdge = -1;

        poly.push_back(vertices[currV]);

        // ---------- 4. 顺着 edge loop 走 ----------
        while (true)
        {
            const auto &edgesAtV = v2edges[currV];

            int nextEdge = -1;
            for (int eIdx : edgesAtV)
            {
                if (eIdx != prevEdge)
                {
                    nextEdge = eIdx;
                    break;
                }
            }

            if (nextEdge < 0)
                break;

            const auto &e = edges[nextEdge];
            int nextV = (e[0] == currV) ? e[1] : e[0];

            // 回到起始点，闭环完成
            if (nextV == startV)
                break;

            poly.push_back(vertices[nextV]);

            prevEdge = nextEdge;
            currV = nextV;
        }

        return poly;
    }

    // =================================

    ChebyshevObject::ChebyshevObject(
        const MatrixXf &sites_,
        OriFun oriFun_,
        AniFun aniFun_,
        float extent,
        int nDim_)
        : sites(sites_), oriFun(oriFun_), aniFun(aniFun_), domainExtent(extent), nDim(nDim_)
    {
        numSites = sites.rows();
        sIdxs.resize(numSites);
        for (int i = 0; i < numSites; ++i)
            sIdxs[i] = i;

        sitesNeighbors.resize(
            numSites,
            std::vector<std::vector<std::pair<int, int>>>(nDim * 2));

        // ===== compute cellScale =====
        float sDiv = std::pow(float(numSites), 1.f / nDim);

        bool isGrid = false;
        float sDivRound = std::round(sDiv);

        if (std::abs(sDiv - sDivRound) < 1e-5f)
        {
            int n = int(sDivRound);

            Eigen::MatrixXf gridPts = chebyshevUtils::generateGridPoints(n, nDim, domainExtent);
            if (gridPts.rows() == sites.rows())
            {
                float meanNorm = 0.f;
                for (int i = 0; i < sites.rows(); ++i)
                    meanNorm += (gridPts.row(i) - sites.row(i)).norm();

                meanNorm /= sites.rows();

                if (meanNorm < 1.f / nDim)
                    isGrid = true;
            }
        }

        // 2 * extent / (sDiv if isGrid else 1)
        cellScale = 2.f * domainExtent / (isGrid ? sDiv : 1.f);
        cellScales.assign(numSites, cellScale);

        baseDir.resize(2 * nDim, nDim);
        baseDir.setZero(); // 先初始化为全0矩阵

        // 核心循环：动态生成每个轴的正负方向向量
        for (int axis = 0; axis < nDim; ++axis)
        {
            // +axis
            baseDir(axis, axis) = 1.0f;

            // -axis
            baseDir(nDim + axis, axis) = -1.0f;
        }

        initCells();
    }

    /* ========================================================= */

    void ChebyshevObject::initCells()
    {
        timings.emplace_back();

        /* ---------- metric matrices Ms ---------- */
        Ms.resize(numSites);

        if (!oriFun)
        {
            for (int i = 0; i < numSites; ++i)
                Ms[i] = Eigen::MatrixXf::Identity(nDim, nDim);
        }
        else
        {
            Eigen::VectorXf angles = oriFun(sites);
            for (int i = 0; i < numSites; ++i)
                Ms[i] = chebyshevUtils::MrDynamic(angles.row(i), nDim);
        }

        /* ---------- Mvecs ---------- */
        Mvecs.resize(numSites);
        for (int i = 0; i < numSites; ++i)
        {
            Eigen::MatrixXf V(2 * nDim, nDim);
            V << Ms[i].transpose(),
                -Ms[i].transpose();
            Mvecs[i] = V;
        }

        /* ---------- anisotropy lambdas ---------- */
        if (!aniFun)
            lambdas = Eigen::MatrixXf::Ones(numSites, nDim * 2);
        else
            lambdas = aniFun(sites);

        /* ---------- domain planes ---------- */
        domainPlanes.resize(nDim * 2);

        for (int d = 0; d < nDim; ++d)
        {
            Eigen::MatrixXf P(2, nDim);
            P.setZero();

            P(0, d) = domainExtent;
            P(1, d) = 1.f;
            domainPlanes[d] = P;

            P(0, d) = -domainExtent;
            P(1, d) = -1.f;
            domainPlanes[d + nDim] = P;
        }

        /* ---------- cut planes ---------- */
        cutPlanes.clear();
        for (int k = 0; k < nDim * 2; ++k)
        {
            long long key = -k;

            Eigen::VectorXf o = domainPlanes[k].row(0);
            Eigen::VectorXf n = domainPlanes[k].row(1);

            cutPlanes[key] = {{o, n}, {-1, -1}};
        }

        cutPlaneKeys.assign(
            numSites,
            std::vector<std::vector<long long>>(nDim * 2));

        /* ---------- cell sectors ---------- */
        cellSectors.resize(numSites);
        for (auto &v : cellSectors)
            v.resize(nDim * 2);

        for (int sIdx = 0; sIdx < numSites; ++sIdx)
        {
            for (int di = 0; di < nDim * 2; ++di)
            {
                // ===== scale = cellScale * lambdas =====
                std::vector<float> scale(nDim * 2);
                for (int k = 0; k < nDim * 2; ++k)
                    scale[k] = cellScales[sIdx] * lambdas(sIdx, k);

                if (nDim == 2)
                {
                    cellSectors[sIdx][di] = std::make_unique<TriCutObject>(sites.row(sIdx).transpose(), di, scale, Ms[sIdx]);

                    // --- domain intersection check ---
                    const auto &verts = cellSectors[sIdx][di]->getVertices();
                    Eigen::Vector2f mins = verts[0];
                    Eigen::Vector2f maxs = verts[0];

                    for (int i = 1; i < verts.size(); ++i)
                    {
                        mins = mins.cwiseMin(verts[i]);
                        maxs = maxs.cwiseMax(verts[i]);
                    }

                    for (int d = 0; d < nDim; ++d)
                    {
                        if (maxs[d] > domainExtent)
                            cutPlaneKeys[sIdx][di].push_back(-d);
                        if (mins[d] < -domainExtent)
                            cutPlaneKeys[sIdx][di].push_back(-(d + nDim));
                    }
                }
                // nDim == 3 → PyraCutObject（之后补）
            }
        }
    }

    /* ========================================================= */

    void ChebyshevObject::computeNeighborsAndPlanes()
    {
        // std::cout << "[computeNeighborsAndPlanes] start\n";

        struct SiteTuple
        {
            int sIdx, di, sJdx, dj;
        };

        using SectorGeometry = std::vector<MatrixXf>;
        std::vector<SiteTuple> tuples;
        std::vector<SectorGeometry> geometries;

        // ---------- 1. 收集候选 ----------
        int candidateCount = 0;
        for (int sIdx = 0; sIdx < numSites; ++sIdx)
        {
            for (int sJdx = sIdx + 1; sJdx < numSites; ++sJdx)
            {
                if (!chebyshevUtils::haveCommonElement({sIdx, sJdx}, sIdxs))
                    continue;
                if ((sites.row(sIdx) - sites.row(sJdx)).norm() > std::sqrt((float)nDim) * 2.0f * domainExtent)
                    continue;

                for (int di = 0; di < nDim * 2; ++di)
                {
                    for (int dj = 0; dj < nDim * 2; ++dj)
                    {
                        auto geom = getSitesSectorGeometry(
                            sIdx, di, sJdx, dj);

                        if (geom.empty())
                            continue;

                        tuples.push_back({sIdx, di, sJdx, dj});
                        geometries.push_back(geom);
                        candidateCount++;
                    }
                }
            }
        }
        // std::cout << "[computeNeighborsAndPlanes] candidate tuples count: " << candidateCount << "\n";

        // ---------- 2. sectorsDoIntersect 过滤 ----------
        std::vector<SiteTuple> validTuples;
        int intersectCount = 0;

        for (size_t i = 0; i < geometries.size(); ++i)
        {
            if (sectorsDoIntersect(geometries[i]))
            {
                validTuples.push_back(tuples[i]);
                intersectCount++;
            }
        }
        // std::cout << "[computeNeighborsAndPlanes] valid tuples after sectorsDoIntersect: " << intersectCount << "\n";

        int N = static_cast<int>(validTuples.size());

        if (N == 0)
        {
            std::cout << "[computeNeighborsAndPlanes] no valid tuples, returning.\n";
            return;
        }

        MatrixXf sitesA(N, nDim);
        MatrixXf sitesB(N, nDim);
        MatrixXf vecsA(N, nDim);
        MatrixXf vecsB(N, nDim);
        Vec lambdasA(N);
        Vec lambdasB(N);

        for (int i = 0; i < N; ++i)
        {
            const auto &t = validTuples[i];
            sitesA.row(i) = sites.row(t.sIdx);
            sitesB.row(i) = sites.row(t.sJdx);
            vecsA.row(i) = Mvecs[t.sIdx].row(t.di);
            vecsB.row(i) = Mvecs[t.sJdx].row(t.dj);
            lambdasA(i) = lambdas(t.sIdx, t.di);
            lambdasB(i) = lambdas(t.sJdx, t.dj);
        }

        // std::cout << std::fixed << std::setprecision(4);
        //  std::cout << "[computeNeighborsAndPlanes] prepared input matrices, N = " << N << "\n";
        //  std::cout << "Sample siteA row 0: ";
        for (int d = 0; d < nDim; ++d)
            std::cout << sitesA(0, d) << (d == nDim - 1 ? "\n" : ", ");

        // ---------- 4. 向量化计算切割平面 ----------
        auto cutPlanesVec = chebyshevUtils::computeCutPlanesVectorized(
            sitesA, sitesB, vecsA, vecsB, lambdasA, lambdasB);

        std::cout << "[computeNeighborsAndPlanes] computed cutPlanesVec size: " << cutPlanesVec.size() << "\n";

        // ---------- 5. 批量调用 addCutPlane ----------
        for (int i = 0; i < N; ++i)
        {
            const auto &t = validTuples[i];
            const auto &[o, n] = cutPlanesVec[i];

            // std::cout << "[CutPlane " << i << "] "
            //           << "Sites: (" << t.sIdx << ", " << t.sJdx << "), "
            //           << "Dims: (" << t.di << ", " << t.dj << ")\n"
            //           << "  Origin: [";
            // for (int d = 0; d < nDim; ++d)
            //     std::cout << o(d) << (d == nDim - 1 ? "]\n" : ", ");
            // std::cout << "  Normal: [";
            // for (int d = 0; d < nDim; ++d)
            //     std::cout << n(d) << (d == nDim - 1 ? "]\n" : ", ");

            addCutPlane(t.sIdx, t.di, t.sJdx, t.dj, o, n);
        }

        std::cout << "[computeNeighborsAndPlanes] done\n";
    }

    void ChebyshevObject::computeCutPlane(
        int sA, int dA,
        int sB, int dB,
        Vec &pO,
        Vec &pN)
    {
        const float eps = 1e-6f;

        Vec siteA = sites.row(sA);
        Vec siteB = sites.row(sB);

        Vec vecA = Mvecs[sA].row(dA).transpose();
        Vec vecB = Mvecs[sB].row(dB).transpose();

        float lambdaA = lambdas(sA, dA);
        float lambdaB = lambdas(sB, dB);

        float adbs = vecA.dot(vecB);
        Vec BtoA = siteA - siteB;

        bool sameDir = std::abs(1.f - adbs) < eps;

        bool samePos = sameDir && std::abs(vecA.dot(BtoA)) < eps;

        bool lambdasMask = sameDir && std::abs(lambdaA - lambdaB) < eps;

        Vec bDiv = vecB * (lambdaA / lambdaB);

        if (lambdasMask)
            bDiv *= -1.f;

        float denom = 1.f - vecA.dot(bDiv);
        if (std::abs(denom) < eps)
        {
            pN.setZero();
            return;
        }
        float t = BtoA.dot(bDiv) / denom;

        pO = siteA + vecA * t;

        pN = vecA / lambdaA - vecB / lambdaB;

        float nrm = pN.norm();
        if (nrm > eps)
            pN /= nrm;
        else
            pN.setZero();

        if (sameDir)
            pN = vecA;

        if (lambdasMask)
            pN.setZero();

        if (samePos)
        {
            pN = BtoA.normalized();

            pO = (siteA * lambdaB + siteB * lambdaA) / (lambdaA + lambdaB);
        }
    }

    /* ========================================================= */

    void ChebyshevObject::addCutPlane(
        int sIdx, int di, int sJdx, int dj,
        const Vec &o,
        const Vec &n)
    {
        sitesNeighbors[sIdx][di].push_back({sJdx, dj});
        sitesNeighbors[sJdx][dj].push_back({sIdx, di});

        if (n.norm() < 1e-6)
            return;

        long long key =
            chebyshevUtils::cantorPi(chebyshevUtils::cantorPiO(sIdx, di),
                                     chebyshevUtils::cantorPiO(sJdx, dj));

        cutPlanes[key] = {{o, n}, {sIdx, sJdx}};
        auto tryAddKey = [&](int s, int d)
        {
            for (auto k : cutPlaneKeys[s][d])
            {
                if (chebyshevUtils::planesEquiv(cutPlanes[k].first,
                                                std::make_pair(o, n),
                                                1e-6f))
                    return;
            }
            cutPlaneKeys[s][d].push_back(key);
        };

        tryAddKey(sIdx, di);
        tryAddKey(sJdx, dj);
    }

    /* ========================================================= */

    void ChebyshevObject::cutWithPlanes()
    {
        for (int sIdx = 0; sIdx < numSites; ++sIdx)
        {
            for (int di = 0; di < nDim * 2; ++di)
            {
                auto &sector = cellSectors[sIdx][di];

                for (auto key : cutPlaneKeys[sIdx][di])
                {
                    auto &plane = cutPlanes[key].first;
                    DBG("=================================================");
                    DBG("[ChebyshevObject::cutWithPlanes]");
                    DBG("  sIdx = " << sIdx << ", di = " << di << ", cutPlaneKey = " << key);
                    DBG("  plane origin o = [" << plane.first.transpose() << "]");
                    DBG("  plane normal n = [" << plane.second.transpose() << "]");
                    DBG("-------------------------------------------------");
                    sector->cutWithPlane(
                        plane.first,
                        plane.second,
                        key);
                }

                sector->computePolysCentroidsAndWeights();
            }
        }
    }

    /* ========================================================= */

    void ChebyshevObject::clipCellGeometry()
    {
        for (int sIdx = 0; sIdx < numSites; ++sIdx)
        {
            for (int di = 0; di < nDim * 2; ++di)
            {
                auto &cellSec = cellSectors[sIdx][di];

                // ---- 收集邻居 index ----
                std::unordered_set<int> neighborSet;
                for (auto &nb : sitesNeighbors[sIdx][di])
                {
                    int sJdx = nb.first;
                    if (sJdx != sIdx)
                        neighborSet.insert(sJdx);
                }

                // ---- 当前 site 参数 ----
                Vec siteA = sites.row(sIdx);
                Vec MvecA = Mvecs[sIdx].row(di);
                float lambdaA = lambdas(sIdx, di);

                // ---- 构造 neighbor 数据 ----
                std::vector<Vec> sitesB;
                std::vector<Eigen::MatrixXf> MvecsB;
                std::vector<Vec> lambdasB;

                for (int nbIdx : neighborSet)
                {
                    sitesB.push_back(sites.row(nbIdx));
                    MvecsB.push_back(Mvecs[nbIdx]); // (2*nDim × nDim)
                    lambdasB.push_back(lambdas.row(nbIdx));
                }

                // ---- 调用函数 ----
                infinityVoronoi::clipCellGeometry(
                    *cellSec,
                    siteA,
                    MvecA,
                    lambdaA,
                    sitesB,
                    MvecsB,
                    lambdasB,
                    domainExtent);
            }
        }
    }

    /* ========================================================= */

    void ChebyshevObject::dissolveCells()
    {

        // std::cout << "[dissolveCell] site num is " << numSites << " start\n";

        std::vector<DissolveResult> dissolveResults;
        dissolveResults.reserve(cellSectors.size());

        int idx = 0;
        for (auto &sectorList : cellSectors)
        {
            // std::cout << "[dissolveCell] sector group " << idx++ << "\n";
            dissolveResults.push_back(dissolve(sectorList, nDim));
        }

        processDissolvedSectors(dissolveResults);

        cellCentroids.resize(numSites, nDim);
        cellBBcenters.resize(numSites, nDim);

        int bbVertCount = 1 << nDim;
        cellBBs.resize(numSites);
        cellAdjacency.resize(numSites);

        for (int sIdx = 0; sIdx < numSites; ++sIdx)
        {
            // std::cout << "  cell " << sIdx << "\n";

            //----------------------------------
            // centroid
            //----------------------------------
            std::vector<Vec> allCentroids;
            std::vector<float> allWeights;

            for (auto &secPtr : cellSectors[sIdx])
            {
                auto centroids = secPtr->getPolysCentroids();
                auto weights = secPtr->getPolysWeights();

                allCentroids.insert(
                    allCentroids.end(),
                    centroids.begin(),
                    centroids.end());

                allWeights.insert(
                    allWeights.end(),
                    weights.begin(),
                    weights.end());
            }

            Vec weightedSum = Vec::Zero(nDim);
            float weightSum = 0.f;

            for (size_t i = 0; i < allCentroids.size(); ++i)
            {
                weightedSum += allCentroids[i] * allWeights[i];
                weightSum += allWeights[i];
            }

            if (weightSum > 0.f)
                cellCentroids.row(sIdx) = weightedSum / weightSum;
            else
                cellCentroids.row(sIdx).setZero();

            //----------------------------------
            // bounding box verts
            //----------------------------------
            std::vector<Vec> allVerts;

            for (auto &poly : cellVertexSets[sIdx])
            {
                allVerts.insert(
                    allVerts.end(),
                    poly.begin(),
                    poly.end());
            }

            if (allVerts.empty())
            {
                cellBBcenters.row(sIdx).setZero();
                cellBBs[sIdx].resize(0, nDim);
                continue;
            }

            MatrixXf vertsMat(allVerts.size(), nDim);
            for (size_t i = 0; i < allVerts.size(); ++i)
            {
                vertsMat.row(i) = allVerts[i];
            }

            MatrixXf cellVerts = vertsMat * Ms[sIdx];

            Vec vMin = cellVerts.colwise().minCoeff();
            Vec vMax = cellVerts.colwise().maxCoeff();

            Vec bbCenter = (vMin + vMax) * 0.5f;

            cellBBcenters.row(sIdx) =
                (bbCenter * Ms[sIdx].transpose()).transpose();

            //----------------------------------
            // BB verts
            //----------------------------------
            cellBBs[sIdx].resize(bbVertCount, nDim);

            Vec halfSize = (vMax - vMin) * 0.5f;

            for (int i = 0; i < bbVertCount; ++i)
            {
                Vec corner = bbCenter;

                for (int d = 0; d < nDim; ++d)
                {
                    float sign = (i & (1 << d)) ? 1.f : -1.f;
                    corner[d] += sign * halfSize[d];
                }

                cellBBs[sIdx].row(i) =
                    (corner * Ms[sIdx].transpose()).transpose();
            }
            //----------------------------------
            // adjacency
            //----------------------------------
            std::unordered_set<int> adjSet;

            for (size_t i = 0; i < cellPlaneKeys[sIdx].size(); ++i)
            {
                long long planeKey = cellPlaneKeys[sIdx][i];

                auto it = cutPlanes.find(planeKey);
                // 新增日志2：打印find结果
                if (it == cutPlanes.end())
                {
                    continue;
                }

                auto &pairSites = it->second.second;

                if (pairSites.first != sIdx)
                {
                    adjSet.insert(pairSites.first);
                }
                if (pairSites.second != sIdx)
                {
                    adjSet.insert(pairSites.second);
                }
            }

            cellAdjacency[sIdx] = std::vector<int>(adjSet.begin(), adjSet.end());
        }

        std::unordered_set<uint64_t> uniqueEdges;
        cellAdjacencyEdges.clear();

        for (int sIdx = 0; sIdx < numSites; ++sIdx)
        {
            for (int aIdx : cellAdjacency[sIdx])
            {
                int a = std::min(sIdx, aIdx);
                int b = std::max(sIdx, aIdx);

                uint64_t key =
                    (uint64_t(a) << 32) | uint32_t(b);

                if (uniqueEdges.insert(key).second)
                    cellAdjacencyEdges.emplace_back(a, b);
            }
        }

        std::cout << "[dissolve] done\n";
    }

    /* ========================================================= */

    void ChebyshevObject::computeDiagram(bool doFinish)
    {
        computeNeighborsAndPlanes();
        std::cout << "neighbors and planes compute finish" << std::endl;
        cutWithPlanes();
        std::cout << "plane cutted compute finish" << std::endl;
        clipCellGeometry();
        std::cout << "clip cell finish" << std::endl;
        dissolveCells();
        std::cout << "dissolve cell finish" << std::endl;

        if (doFinish)
            finish();
    }

    /* ========================================================= */

    void ChebyshevObject::lloydRelax(float itersThresh)
    {
        if (cellCentroids.rows() == 0)
            computeDiagram(false);

        int iter = 0;
        while (true)
        {
            Vec move = (sites - cellCentroids).rowwise().norm();

            if ((itersThresh < 1.0f &&
                 move.maxCoeff() < itersThresh) ||
                iter == (int)itersThresh)
                break;

            sites = cellCentroids;
            initCells();
            computeDiagram(false);
            ++iter;
        }
        finish();
    }

    /* ========================================================= */

    void ChebyshevObject::finish()
    {
        logMeta();
    }

    ChebyshevObject2D::ChebyshevObject2D(
        const MatrixXf &sites,
        OriFun oriFun,
        AniFun aniFun,
        float extent)
        : ChebyshevObject(
              sites,
              oriFun,
              aniFun,
              extent,
              2) // <<< 强制2D
    {
    }

    /* =======================================================
   meta info
======================================================= */

    void ChebyshevObject2D::logMeta()
    {
        if (cellAdjacency.empty())
            return;

        int minAdj = INT_MAX, maxAdj = 0;
        float medAdj = 0.f;

        std::vector<int> counts;
        counts.reserve(cellAdjacency.size());

        for (auto &adj : cellAdjacency)
        {
            int n = (int)adj.size();
            counts.push_back(n);
            minAdj = std::min(minAdj, n);
            maxAdj = std::max(maxAdj, n);
        }

        std::sort(counts.begin(), counts.end());
        medAdj = counts[counts.size() / 2];

        std::cout
            << "#/Cell Adj: "
            << minAdj << " / "
            << maxAdj << " / "
            << medAdj << std::endl;
    }

    /* =======================================================
   sector geometry
======================================================= */

    std::vector<MatrixXf>
    ChebyshevObject2D::getSitesSectorGeometry(
        int sIdx, int di,
        int sJdx, int dj)
    {
        if (di >= cellSectors[sIdx].size() ||
            dj >= cellSectors[sJdx].size())
            return {};
        MatrixXf A = vecArrayToMatrix(cellSectors[sIdx][di]->getVertices());
        MatrixXf B = vecArrayToMatrix(cellSectors[sJdx][dj]->getVertices());

        return {A, B};
    }

    /* =======================================================
       dissolve result unpack
    ======================================================= */

    void ChebyshevObject2D::processDissolvedSectors(
        const std::vector<DissolveResult> &dData)
    {
        int n = (int)dData.size();

        // std::cout << "[processDissolved] start, cell count = "
        //           << n << std::endl;

        cellVertexSets.resize(n);
        cellPlaneKeys.resize(n);

        for (int i = 0; i < n; ++i)
        {
            const auto &polys = dData[i].polys;
            const auto &pKeys = dData[i].planeKeys;

            // std::cout << "[processDissolved]  cell " << i
            //           << " polys = " << polys.size()
            //           << " planeKeys = " << pKeys.size()
            //           << std::endl;

            // ---- 检查空 cell ----
            if (polys.empty())
            {
                std::cout << "[processDissolved]   WARNING: empty polys after dissolve\n";
            }

            // ---- 检查 polygon ----
            // for (size_t p = 0; p < polys.size(); ++p)
            // {
            //     const auto &poly = polys[p];

            //     if (poly.empty())
            //     {
            //         std::cout << "[processDissolved]   WARNING: poly "
            //                   << p << " has 0 vertices\n";
            //         continue;
            //     }

            //     if (poly.size() < 3)
            //     {
            //         std::cout << "[processDissolved]   WARNING: poly "
            //                   << p << " vertex count = "
            //                   << poly.size()
            //                   << " (<3)\n";
            //     }

            //     // optional: 检查维度
            //     // if (poly[0].size() != nDim)
            //     // {
            //     //     std::cout << "[processDissolved]   ERROR: poly dim mismatch: "
            //     //               << poly[0].size()
            //     //               << " vs nDim=" << nDim
            //     //               << std::endl;
            //     // }
            // }

            // ---- 真正赋值 ----
            cellVertexSets[i] = polys;
            cellPlaneKeys[i] = pKeys;
        }

        // std::cout << "[processDissolved] done\n";
    }

    void ChebyshevObject::debugPrintSitesAndLambdas() const
    {
        std::cout << "==== Chebyshev Debug: Sites & Lambdas ====\n";
        std::cout << "numSites = " << sites.rows()
                  << ", nDim = " << nDim << "\n";

        for (int i = 0; i < sites.rows(); i++)
        {
            Vec site = sites.row(i).transpose();
            Vec lambda = lambdas.row(i).transpose();

            std::cout << "[Site " << i << "] ";
            std::cout << "site = ("
                      << site.x() << ", "
                      << site.y();
            if (nDim == 3)
                std::cout << ", " << site.z();
            std::cout << "), lambdas = ";

            for (int j = 0; j < 2 * nDim; j++)
            {
                std::cout << lambda[j];
                if (j + 1 < 2 * nDim)
                    std::cout << ", ";
            }
            std::cout << "\n";
        }
    }

    void ChebyshevObject::drawSiteWithDirs(Color c, float thickness, float scale) const
    {
        for (int i = 0; i < sites.rows(); i++)
        {
            const Vec &site = sites.row(i).transpose();
            const MatrixXf &rot = Ms[i];
            float z = nDim == 2 ? 0 : site.z();
            const Vec &lambda = lambdas.row(i).transpose();
            for (int j = 0; j < 2 * nDim; j++)
            {
                Vec dir = baseDir.row(j).transpose();
                Vec rotEnd = site + rot.transpose() * dir * scale * lambda[j];
                DrawCylinderEx({site.x(), z, -site.y()}, {rotEnd.x(), z, -rotEnd.y()}, thickness, thickness, 1, c);
            }
        }
    }

    void ChebyshevObject::drawCutPlanes(Color c, float thickness, float extendScale) const
    {
        float L = domainExtent * extendScale;
        // std::cout << "[drawCutPlanes] total planes: " << cutPlanes.size() << std::endl;

        for (const auto &[key, planeData] : cutPlanes)
        {
            const Vec &o = planeData.first.first;
            const Vec &n = planeData.first.second;

            // std::cout << "Plane key: " << key << "\n";
            // std::cout << "  Origin: [";
            // for (int i = 0; i < nDim; ++i)
            //     std::cout << o(i) << (i == nDim - 1 ? "]\n" : ", ");
            // std::cout << "  Normal: [";
            // for (int i = 0; i < nDim; ++i)
            //     std::cout << n(i) << (i == nDim - 1 ? "]\n" : ", ");

            float norm_n = n.norm();
            // std::cout << "  Normal norm: " << norm_n << "\n";

            // 如果法线太小，也打印提示
            if (norm_n < 1e-6f)
            {
                // std::cout << "  Warning: normal norm too small, skipping drawing direction calculation.\n";
                continue;
            }

            // 2D: direction = perpendicular to normal
            Vec t(2);
            t << -n(1), n(0);
            t.normalize();

            Vec p0 = o - t * L;
            Vec p1 = o + t * L;

            // std::cout << "  Drawing line from p0: [" << p0.x() << ", " << p0.y() << "] to p1: [" << p1.x() << ", " << p1.y() << "]\n";

            DrawCylinderEx(
                {p0.x(), 0.f, -p0.y()},
                {p1.x(), 0.f, -p1.y()},
                thickness,
                thickness,
                1,
                c);

            // --- optional: draw normal ---
            Vec pn = o + n;

            // std::cout << "  Drawing normal from o to pn: [" << pn.x() << ", " << pn.y() << "]\n";

            DrawCylinderEx(
                {o.x(), 0.f, -o.y()},
                {pn.x(), 0.f, -pn.y()},
                thickness * 0.6f,
                thickness * 1.2f,
                1,
                RL_BLUE);
        }
    }

    void ChebyshevObject::drawCellSectors(float thickness) const
    {
        std::vector<polyloop::Polyloop2> cuttedPolys;
        for (const auto &siteSectors : cellSectors)
        {
            // const auto &siteSectors = cellSectors[0];
            
            for (const auto &sectorPtr : siteSectors)
            {
                if (!sectorPtr)
                    continue;

                const SectorCutObject &sector = *sectorPtr;

                for (const auto &[k, _] : sector.polys)
                {
                    cuttedPolys.emplace_back(sector.buildPolyFromKey(k));
                }
              
            }
        }
        for (int i = 0; i < cuttedPolys.size(); i++)
        {
            const auto &cell = cuttedPolys[i];
            Color c = renderUtil::ColorFromHue(float(i) / cuttedPolys.size());
            render::stroke_bold_polygon2(cell, c, 0.f, thickness, c.a);
            render::fill_polygon2(cell, c, 0.f, 0.13f);
        }
    }

    void ChebyshevObject::drawCellSector(
        int sIdx,
        int di,
        float thickness) const
    {
        // ---- safety check ----
        if (sIdx < 0 || sIdx >= cellSectors.size())
            return;
        if (di < 0 || di >= cellSectors[sIdx].size())
            return;

        const auto &sectorPtr = cellSectors[sIdx][di];
        if (!sectorPtr)
            return;

        const SectorCutObject &sector = *sectorPtr;

        std::vector<polyloop::Polyloop2> polys;
        polys.reserve(sector.polys.size());

        // ---- collect polys ----
        for (const auto &[k, _] : sector.polys)
        {
            polys.emplace_back(sector.buildPolyFromKey(k));
        }

        // ---- draw ----
        for (int i = 0; i < polys.size(); ++i)
        {
            const auto &cell = polys[i];

            // 颜色可以按 poly index，便于区分 split 结果
            Color c = renderUtil::ColorFromHue(
                polys.size() > 1 ? float(i) / polys.size() : 0.0f);

            render::stroke_bold_polygon2(cell, c, 0.f, thickness, c.a);
            render::fill_polygon2(cell, c, 0.f, 0.13f);
        }
    }
}