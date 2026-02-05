#include "chebyshevUtils.h"

namespace chebyshevUtils
{
    using Vec = Eigen::VectorXf;
    using Vec2 = Eigen::Vector2f;
    using namespace Eigen;
    // edges: vector of {v0, v1}
    std::optional<PathResult> edgesToPath(
        const std::vector<std::array<int, 2>> &edgesIn,
        bool withClosedFlag,
        bool returnPartial)
    {
        // deep copy
        std::vector<std::array<int, 2>> edges = edgesIn;
        if (edges.empty())
            return std::nullopt;

        // init path
        std::vector<int> path = {edges[0][0], edges[0][1]};
        edges.erase(edges.begin());

        int iters = 0;

        while (!edges.empty())
        {
            auto edge = edges.front();
            edges.erase(edges.begin());

            bool attached = true;

            if (path.front() == edge[0])
                path.insert(path.begin(), edge[1]);
            else if (path.back() == edge[0])
                path.push_back(edge[1]);
            else if (path.front() == edge[1])
                path.insert(path.begin(), edge[0]);
            else if (path.back() == edge[1])
                path.push_back(edge[0]);
            else
            {
                // cannot attach
                edges.push_back(edge);
                attached = false;
                iters++;
            }

            if (attached)
                iters = 0;

            // no progress in one full loop
            if (!edges.empty() && iters > (int)edges.size())
            {
                if (returnPartial)
                    break;
                else
                    return std::nullopt;
            }
        }

        bool closed = (!path.empty() && path.front() == path.back());

        if (closed)
            path.pop_back();

        if (withClosedFlag)
            return PathResult{path, closed};
        else
            return PathResult{path, false};
    }

    std::vector<std::vector<int>> edgesToPaths(const std::vector<Eigen::Vector2i> &edges, std::vector<bool> *closedFlags)
    {
        std::unordered_map<int, std::vector<int>> adj;
        for (auto &e : edges)
        {
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }

        std::unordered_set<int> visited;
        std::vector<std::vector<int>> paths;
        if (closedFlags)
            closedFlags->clear();

        for (auto &[v, _] : adj)
        {
            if (visited.count(v))
                continue;

            auto &n = adj[v];
            std::vector<int> path;
            bool closed = false;

            if (n.size() == 2)
                path = {n[0], v, n[1]};
            else
                path = {n[0], v};

            bool extend[2] = {true, true};

            while ((extend[0] || extend[1]) && !closed)
            {
                // front
                if (extend[0])
                {
                    int cur = path.front();
                    int prev = path[1];
                    auto &ns = adj[cur];
                    if (ns.size() == 1 && ns[0] == prev)
                        extend[0] = false;
                    else
                    {
                        int next = (ns[0] != prev) ? ns[0] : ns[1];
                        if (next == path.back())
                            closed = true;
                        else
                            path.insert(path.begin(), next);
                    }
                }

                // back
                if (extend[1] && !closed)
                {
                    int cur = path.back();
                    int prev = path[path.size() - 2];
                    auto &ns = adj[cur];
                    if (ns.size() == 1 && ns[0] == prev)
                        extend[1] = false;
                    else
                    {
                        int next = (ns[0] != prev) ? ns[0] : ns[1];
                        if (next == path.front())
                            closed = true;
                        else
                            path.push_back(next);
                    }
                }
            }

            for (int vtx : path)
                visited.insert(vtx);
            paths.push_back(path);
            if (closedFlags)
                closedFlags->push_back(closed);
        }

        return paths;
    }

    std::vector<std::vector<Eigen::Vector2i>>
    findConnectedEdgeSegments(const std::vector<Eigen::Vector2i> &edges)
    {
        std::vector<std::vector<Eigen::Vector2i>> segs;
        for (auto &e : edges)
            segs.push_back({e});

        bool merged = true;
        while (merged)
        {
            merged = false;
            for (size_t i = 0; i < segs.size(); ++i)
            {
                for (size_t j = i + 1; j < segs.size(); ++j)
                {
                    std::unordered_set<int> vi, vj;
                    for (auto &e : segs[i])
                    {
                        vi.insert(e[0]);
                        vi.insert(e[1]);
                    }
                    for (auto &e : segs[j])
                    {
                        vj.insert(e[0]);
                        vj.insert(e[1]);
                    }

                    bool intersect = false;
                    for (int v : vi)
                        if (vj.count(v))
                        {
                            intersect = true;
                            break;
                        }

                    if (intersect)
                    {
                        segs[j].insert(segs[j].end(), segs[i].begin(), segs[i].end());
                        segs.erase(segs.begin() + i);
                        merged = true;
                        break;
                    }
                }
                if (merged)
                    break;
            }
        }
        return segs;
    }

    std::vector<Eigen::Vector2f>
    intersectLinesLine2D(
        const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> &ons,
        const Eigen::Vector2f &o,
        const Eigen::Vector2f &n)
    {
        std::vector<Eigen::Vector2f> res;
        res.reserve(ons.size());

        Eigen::Vector2f ortho_n = geo::rotate90CW(n);

        for (const auto &[p, dir] : ons)
        {
            float denom = dir.dot(ortho_n);
            if (std::abs(denom) < 1e-6f)
                continue; //

            float s = (p - o).dot(n) / denom;
            res.push_back(p + geo::rotate90CW(dir) * s);
        }
        return res;
    }

    std::pair<Eigen::Vector2f, float>
    computePolygonCentroid2D(const std::vector<Eigen::Vector2f> &pts, bool withArea)
    {
        const int n = (int)pts.size();
        if (n == 0)
            return {Eigen::Vector2f::Zero(), 0.0f};

        float area2 = 0.0f; // 2 * area
        Eigen::Vector2f centroid = Eigen::Vector2f::Zero();

        for (int i = 0; i < n; ++i)
        {
            const Eigen::Vector2f &p = pts[i];
            const Eigen::Vector2f &r = pts[(i - 1 + n) % n]; // roll by 1

            float w = p.x() * r.y() - r.x() * p.y();
            area2 += w;
            centroid += (p + r) * w;
        }

        float area = area2 * 0.5f;

        if (std::abs(area) > 1e-8f)
        {
            centroid /= (6.0f * area);
        }
        else
        {
            // degenerate polygon → fallback to mean
            centroid.setZero();
            for (const auto &p : pts)
                centroid += p;
            centroid /= (float)n;
        }

        return withArea
                   ? std::make_pair(centroid, std::abs(area))
                   : std::make_pair(centroid, 0.0f);
    }

    std::vector<std::vector<Vec2>> concatPolyParts(const std::vector<std::vector<Vec2>> &polyParts)
    {
        // 存储最终拼接后的完整多边形
        std::vector<std::vector<Vec2>> polys;

        // 遍历每个待拼接的多边形片段
        for (const auto &part : polyParts)
        {
            // 标记当前片段是否已完成拼接
            bool merged = false;

            // 遍历已有的完整多边形，尝试拼接当前片段
            for (auto &poly : polys)
            {
                // 提取当前片段和已有多边形的首尾顶点（二维向量）
                const Vec2 &partHead = part.front(); // 片段首顶点
                const Vec2 &partTail = part.back();  // 片段尾顶点
                const Vec2 &polyHead = poly.front(); // 已有多边形首顶点
                const Vec2 &polyTail = poly.back();  // 已有多边形尾顶点

                // 拼接方式1：片段尾 → 多边形首（直接拼接，无需反转）
                if ((partTail - polyHead).squaredNorm() < 1e-8f)
                {
                    std::vector<Vec2> mergedPoly = part;
                    // 将已有多边形的所有顶点追加到片段末尾
                    mergedPoly.insert(mergedPoly.end(), poly.begin(), poly.end());
                    poly = std::move(mergedPoly); // 移动语义避免拷贝
                    merged = true;
                    break;
                }

                // 拼接方式2：片段首 → 多边形首（反转片段后拼接）
                if ((partHead - polyHead).squaredNorm() < 1e-8f)
                {
                    // 反转片段顶点顺序（rbegin/rend是反向迭代器）
                    std::vector<Vec2> revPart(part.rbegin(), part.rend());
                    // 追加已有多边形顶点
                    revPart.insert(revPart.end(), poly.begin(), poly.end());
                    poly = std::move(revPart);
                    merged = true;
                    break;
                }

                // 拼接方式3：片段首 → 多边形尾（直接追加到多边形末尾）
                if ((partHead - polyTail).squaredNorm() < 1e-8f)
                {
                    poly.insert(poly.end(), part.begin(), part.end());
                    merged = true;
                    break;
                }

                // 拼接方式4：片段尾 → 多边形尾（反转片段后追加）
                if ((partTail - polyTail).squaredNorm() < 1e-8f)
                {
                    std::vector<Vec2> revPart(part.rbegin(), part.rend());
                    poly.insert(poly.end(), revPart.begin(), revPart.end());
                    merged = true;
                    break;
                }
            }

            // 若当前片段未拼接成功，作为新多边形加入结果
            if (!merged)
                polys.push_back(part);
        }

        return polys;
    }

    std::vector<std::vector<Vec>> concatPolyParts(const std::vector<std::vector<Vec>> &polyParts)
    {
        std::vector<std::vector<Vec>> polys;

        for (const auto &part : polyParts)
        {
            bool merged = false;

            for (auto &poly : polys)
            {
                const Vec &partHead = part.front();
                const Vec &partTail = part.back();
                const Vec &polyHead = poly.front();
                const Vec &polyTail = poly.back();

                // part.tail -> poly.head
                if ((partTail - polyHead).squaredNorm() < 1e-8f)
                {
                    std::vector<Vec> mergedPoly = part;
                    mergedPoly.insert(
                        mergedPoly.end(),
                        poly.begin(), poly.end());
                    poly = std::move(mergedPoly);
                    merged = true;
                    break;
                }

                // part.head -> poly.head
                if ((partHead - polyHead).squaredNorm() < 1e-8f)
                {
                    std::vector<Vec> revPart(part.rbegin(), part.rend());
                    revPart.insert(
                        revPart.end(),
                        poly.begin(), poly.end());
                    poly = std::move(revPart);
                    merged = true;
                    break;
                }

                // part.head -> poly.tail
                if ((partHead - polyTail).squaredNorm() < 1e-8f)
                {
                    poly.insert(poly.end(), part.begin(), part.end());
                    merged = true;
                    break;
                }

                // part.tail -> poly.tail
                if ((partTail - polyTail).squaredNorm() < 1e-8f)
                {
                    std::vector<Vec> revPart(part.rbegin(), part.rend());
                    poly.insert(poly.end(), revPart.begin(), revPart.end());
                    merged = true;
                    break;
                }
            }

            if (!merged)
                polys.push_back(part);
        }

        return polys;
    }

    std::vector<Vec> limitedDissolve2D(const std::vector<Vec> &verts)
    {
        int n = (int)verts.size();
        if (n <= 2)
            return verts;

        std::vector<int> keepIdxs;

        for (int vIdx = 0; vIdx < n; ++vIdx)
        {
            int pIdx = (vIdx - 1 + n) % n;
            int nIdx = (vIdx + 1) % n;

            if ((verts[vIdx] - verts[nIdx]).squaredNorm() < 1e-8f)
                continue;

            Vec v1 = (verts[pIdx] - verts[vIdx]).normalized();
            Vec v2 = (verts[nIdx] - verts[vIdx]).normalized();

            float d = std::abs(v1.dot(v2));

            // 非直线点
            if (d < (1.f - 1e-8f))
                keepIdxs.push_back(vIdx);
        }

        std::vector<Vec> newVerts;
        newVerts.reserve(keepIdxs.size());

        for (int idx : keepIdxs)
            newVerts.push_back(verts[idx]);

        // Python: if len(vIdxs) < n: recurse
        if ((int)newVerts.size() < n)
            return limitedDissolve2D(newVerts);

        return newVerts;
    }

    /*-------------------------------------------------------
 randomJitter: uniform noise in [-scale, scale]
-------------------------------------------------------*/
    Eigen::MatrixXf randomJitter(int count, int dim, float scale)
    {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-scale, scale);

        Eigen::MatrixXf jitter(count, dim);
        for (int i = 0; i < count; ++i)
            for (int d = 0; d < dim; ++d)
                jitter(i, d) = dist(rng);

        return jitter;
    }

    /*-------------------------------------------------------
     generateGridPoints
     d:demension
     returns (n^d) x d matrix
    -------------------------------------------------------*/
    Eigen::MatrixXf generateGridPoints(int n, int d, float e)
    {
        // ptsGrid = linspace(-e, e, n, endpoint=False) + e/n
        Eigen::VectorXf ptsGrid(n);
        float step = (2.f * e) / n;

        for (int i = 0; i < n; ++i)
            ptsGrid[i] = -e + step * i + e / n;

        if (d == 1)
        {
            Eigen::MatrixXf result(n, 1);
            result.col(0) = ptsGrid;
            return result;
        }
        else if (d == 2)
        {
            Eigen::MatrixXf result(n * n, 2);
            int idx = 0;

            for (int y = 0; y < n; ++y)
                for (int x = 0; x < n; ++x)
                {
                    result(idx, 0) = ptsGrid[x];
                    result(idx, 1) = ptsGrid[y];
                    ++idx;
                }

            return result;
        }
        else if (d == 3)
        {
            Eigen::MatrixXf result(n * n * n, 3);
            int idx = 0;

            for (int z = 0; z < n; ++z)
                for (int y = 0; y < n; ++y)
                    for (int x = 0; x < n; ++x)
                    {
                        result(idx, 0) = ptsGrid[x];
                        result(idx, 1) = ptsGrid[y];
                        result(idx, 2) = ptsGrid[z];
                        ++idx;
                    }

            return result;
        }

        throw std::runtime_error("generateGridPoints: dimension not supported");
    }

    /*-------------------------------------------------------
     generateJitteredGridPoints
    -------------------------------------------------------*/
    Eigen::MatrixXf generateJitteredGridPoints(int n, int d, float e)
    {
        Eigen::MatrixXf grid = generateGridPoints(n, d, e);
        Eigen::MatrixXf jitter = randomJitter(
            static_cast<int>(std::pow(n, d)),
            d,
            e / n);

        return grid + jitter;
    }

    std::vector<std::pair<int, int>> computeFaceCutIdxs(const std::vector<std::vector<int>> &faceMasks)
    {
        using Face = std::vector<int>;
        using FaceMask = std::vector<bool>;
        using FaceMasks = std::vector<FaceMask>;
        using IndexPair = std::pair<int, int>;

        const int numFaces = static_cast<int>(faceMasks.size());
        assert(numFaces > 0);

        // --------------------------------------------------
        // 1. concatenate masks
        // --------------------------------------------------
        std::vector<bool> cat;
        std::vector<int> offsets;
        offsets.reserve(numFaces + 1);
        offsets.push_back(0);

        for (const auto &fm : faceMasks)
        {
            cat.insert(cat.end(), fm.begin(), fm.end());
            offsets.push_back(static_cast<int>(cat.size()));
        }

        // --------------------------------------------------
        // 2. detect XOR transitions (True <-> False)
        // --------------------------------------------------
        std::set<int> inFaceIdxs;
        for (int i = 0; i + 1 < (int)cat.size(); ++i)
        {
            if (cat[i] != cat[i + 1])
                inFaceIdxs.insert(i + 1);
        }

        // --------------------------------------------------
        // 3. first/last mask of each face
        // --------------------------------------------------
        std::vector<bool> first(numFaces), last(numFaces);
        for (int f = 0; f < numFaces; ++f)
        {
            first[f] = cat[offsets[f]];
            last[f] = cat[offsets[f + 1] - 1];
        }

        // --------------------------------------------------
        // 4. trueStartIdxs: exactly one of first/last is true
        // --------------------------------------------------
        std::set<int> trueStartIdxs;
        for (int f = 0; f < numFaces; ++f)
        {
            if (first[f] ^ last[f])
                trueStartIdxs.insert(offsets[f]);
        }

        // --------------------------------------------------
        // 5. falseStartIdxs: remove fake transitions across faces
        // --------------------------------------------------
        std::set<int> falseStartIdxs;
        for (int f = 0; f + 1 < numFaces; ++f)
        {
            if (last[f] ^ first[f + 1])
                falseStartIdxs.insert(offsets[f + 1]);
        }

        // --------------------------------------------------
        // 6. merge indices
        // --------------------------------------------------
        std::set<int> resIdxs = inFaceIdxs;
        for (int i : falseStartIdxs)
            resIdxs.erase(i);
        for (int i : trueStartIdxs)
            resIdxs.insert(i);

        // --------------------------------------------------
        // 7. convert global idx -> per-face local idx pairs
        // --------------------------------------------------
        std::vector<IndexPair> result;
        result.reserve(numFaces);

        auto it = resIdxs.begin();
        for (int f = 0; f < numFaces; ++f)
        {
            if (it == resIdxs.end())
                break;

            int g0 = *it++;
            int g1 = *it++;

            result.emplace_back(
                g0 - offsets[f],
                g1 - offsets[f]);
        }

        return result;
    }

    bool haveCommonElement(
        const std::vector<int> &a,
        const std::vector<int> &b)
    {
        const std::vector<int> *pa = &a;
        const std::vector<int> *pb = &b;
        if (b.size() < a.size())
            std::swap(pa, pb);

        for (int x : *pa)
            if (std::find(pb->begin(), pb->end(), x) != pb->end())
                return true;

        return false;
    }

    bool vecsParallel(
        const Vec &u,
        const Vec &v,
        float eps,
        bool signedParallel)
    {
        float d = u.dot(v);
        if (signedParallel)
            return std::abs(1.f - d) < eps;
        else
            return std::abs(1.f - std::abs(d)) < eps;
    }

    bool planesEquiv(
        const std::pair<Vec, Vec> &onA,
        const std::pair<Vec, Vec> &onB,
        float eps)
    {
        return distPointToPlane(onA.first, onB.first, onB.second) < eps &&
               vecsParallel(onA.second, onB.second, eps);
    }

    VectorXf inner1d(const MatrixXf &A, const MatrixXf &B)
    {
        // 假设A,B大小相同 (N x dim)
        assert(A.rows() == B.rows());
        return (A.array() * B.array()).rowwise().sum();
    }

    // 法线归一化，类似 numpy normVec
    MatrixXf normVec(const MatrixXf &vecs)
    {
        MatrixXf normalized = vecs;
        for (int i = 0; i < vecs.rows(); ++i)
        {
            float normVal = vecs.row(i).norm();
            if (normVal > 1e-6)
                normalized.row(i) /= normVal;
            else
                normalized.row(i).setZero();
        }
        return normalized;
    }

    // 向量化 computeCutPlanes，输入是多个向量组
    // sitesA, sitesB: N x dim
    // vecsA, vecsB: N x dim
    // lambdasA, lambdasB: N x 1 (列向量)
    // 返回：N x dim x 2 （起点和法线）
    std::vector<std::pair<VectorXf, VectorXf>> computeCutPlanesVectorized(
        const MatrixXf &sitesA,
        const MatrixXf &sitesB,
        const MatrixXf &vecsA,
        const MatrixXf &vecsB,
        const VectorXf &lambdasA,
        const VectorXf &lambdasB)
    {
        int N = sitesA.rows();
        int dim = sitesA.cols();

        std::vector<std::pair<VectorXf, VectorXf>> results;
        results.reserve(N);

        // 逐行计算 adbs = dot(vecsA, vecsB)
        VectorXf adbs(N);
        for (int i = 0; i < N; ++i)
            adbs(i) = vecsA.row(i).dot(vecsB.row(i));

        // BtoA = sitesA - sitesB
        MatrixXf BtoA = sitesA - sitesB;

        // mask 计算
        std::vector<bool> sameDirMask(N), samePosMask(N), lambdasMask(N);

        for (int i = 0; i < N; ++i)
        {
            sameDirMask[i] = std::abs(1.f - adbs(i)) < 1e-6;
            samePosMask[i] = sameDirMask[i] && (std::abs(vecsA.row(i).dot(BtoA.row(i))) < 1e-6);
            lambdasMask[i] = sameDirMask[i] && (std::abs(lambdasA(i) - lambdasB(i)) < 1e-6);
        }

        // 计算 bDiv = vecsB * (lambdaA / lambdaB)
        MatrixXf bDiv(N, dim);
        for (int i = 0; i < N; ++i)
        {
            float scale = lambdasA(i) / lambdasB(i);
            bDiv.row(i) = vecsB.row(i) * scale;
            if (lambdasMask[i])
                bDiv.row(i) *= -1.f;
        }

        // 计算 denom 和 t
        VectorXf denom(N);
        VectorXf t(N);
        for (int i = 0; i < N; ++i)
        {
            denom(i) = 1.f - vecsA.row(i).dot(bDiv.row(i));
            if (std::abs(denom(i)) < 1e-6)
            {
                // 特殊处理，下面统一处理，t置0
                denom(i) = 1e-6;
                t(i) = 0.f;
            }
            else
            {
                t(i) = BtoA.row(i).dot(bDiv.row(i)) / denom(i);
            }
        }

        // 计算 pOs = sitesA + vecsA * t
        MatrixXf pOs(N, dim);
        for (int i = 0; i < N; ++i)
        {
            pOs.row(i) = sitesA.row(i) + vecsA.row(i) * t(i);
        }

        // 计算 pNs = normVec(vecsA / lambdasA - vecsB / lambdasB)
        MatrixXf diffVec(N, dim);
        for (int i = 0; i < N; ++i)
        {
            diffVec.row(i) = vecsA.row(i) / lambdasA(i) - vecsB.row(i) / lambdasB(i);
        }
        MatrixXf pNs = normVec(diffVec);

        // 根据掩码调整 pNs 和 pOs
        for (int i = 0; i < N; ++i)
        {
            if (sameDirMask[i])
                pNs.row(i) = vecsA.row(i);

            if (lambdasMask[i])
                pNs.row(i).setZero();

            if (samePosMask[i])
            {
                // pNs = normalized(BtoA)
                float normVal = BtoA.row(i).norm();
                if (normVal > 1e-6)
                    pNs.row(i) = BtoA.row(i) / normVal;
                else
                    pNs.row(i).setZero();

                // pOs = weighted average
                float la = lambdasA(i);
                float lb = lambdasB(i);
                pOs.row(i) = (sitesA.row(i) * lb + sitesB.row(i) * la) / (la + lb);
            }
        }

        // 返回
        for (int i = 0; i < N; ++i)
        {
            VectorXf origin = pOs.row(i);
            VectorXf normal = pNs.row(i);
            results.emplace_back(origin, normal);
        }

        return results;
    }
}