#pragma once
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <set>
#include <utility>
#include "geo.h"

namespace chebyshevUtils
{
    inline int simpleSign(float x, float thresh = -1.0f)
    {
        if (thresh >= 0.0f && std::abs(x) <= thresh)
            return 0;
        return (x > 0.f) - (x < 0.f);
    }

    inline Eigen::VectorXi simpleSign(const Eigen::VectorXf &xs, float thresh = -1.0f)
    {
        Eigen::VectorXi res(xs.size());
        for (int i = 0; i < xs.size(); ++i)
            res[i] = simpleSign(xs[i], thresh);
        return res;
    }

    inline uint64_t cantorPair(uint64_t k1, uint64_t k2)
    {
        uint64_t s = k1 + k2;
        return ((s * s + s) >> 1) + k2;
    }

    /*
     *undirectional cantor pairing
     */
    inline uint64_t cantorPi(uint64_t k1, uint64_t k2)
    {
        uint64_t s = k1 + k2;
        return (s * (s + 1)) / 2 + (k1 > k2 ? k1 : k2);
    }

    /*
     * ordered cantor pairing
     */
    inline uint64_t cantorPiO(uint64_t k1, uint64_t k2)
    {
        uint64_t s = k1 + k2;
        return (s * (s + 1)) / 2 + k2;
    }

    inline std::vector<uint64_t> cantorPiV(
        const std::vector<Eigen::Vector2i> &edges,
        bool sort = true)
    {
        std::vector<uint64_t> res;
        res.reserve(edges.size());

        for (auto e : edges)
        {
            uint64_t a = e[0], b = e[1];
            if (sort && a > b)
                std::swap(a, b);
            res.push_back(cantorPair(a, b));
        }
        return res;
    }

    inline Eigen::Matrix2f Mr2D(float a)
    {
        float c = std::cos(a);
        float s = std::sin(a);

        Eigen::Matrix2f R;
        R << c, -s,
            s, c;

        return R;
    }

    inline Eigen::Matrix3f Mr3D(float alpha = 0.f,
                                float beta = 0.f,
                                float gamma = 0.f)
    {
        float ca = std::cos(alpha);
        float sa = std::sin(alpha);
        float cb = std::cos(beta);
        float sb = std::sin(beta);
        float cg = std::cos(gamma);
        float sg = std::sin(gamma);

        Eigen::Matrix3f Rx, Ry, Rz;

        Rx << 1, 0, 0,
            0, ca, -sa,
            0, sa, ca;

        Ry << cb, 0, -sb,
            0, 1, 0,
            sb, 0, cb;

        Rz << cg, -sg, 0,
            sg, cg, 0,
            0, 0, 1;

        return Rx * Ry * Rz; // 顺序必须这样
    }

    inline Eigen::Matrix2f Mr(float angle)
    {
        return Mr2D(angle);
    }

    inline Eigen::Matrix3f Mr(const Eigen::Vector3f &ori)
    {
        return Mr3D(ori.x(), ori.y(), ori.z());
    }

    inline Eigen::MatrixXf MrDynamic(const Eigen::VectorXf &ori, int nDim)
    {
        if (nDim == 2)
            return Mr2D(ori[0]);
        else
            return Mr3D(ori[0], ori[1], ori[2]);
    }

    inline std::vector<int> flatten(const std::vector<Eigen::Vector2i> &edges)
    {
        std::vector<int> out;
        for (auto &e : edges)
        {
            out.push_back(e[0]);
            out.push_back(e[1]);
        }
        return out;
    }

    inline std::vector<std::array<int, 2>> convertVector2array(const std::vector<Eigen::Vector2i> &viArrays)
    {
        std::vector<std::array<int, 2>> intArrays;
        for (const auto &vi : viArrays)
            intArrays.push_back({vi.x(), vi.y()});
        return intArrays;
    }

    struct PathResult
    {
        std::vector<int> path;
        bool closed;
    };

    std::optional<PathResult> edgesToPath(
        const std::vector<std::array<int, 2>> &edgesIn,
        bool withClosedFlag = false,
        bool returnPartial = false);
    std::vector<std::vector<int>> edgesToPaths(const std::vector<Eigen::Vector2i> &edges, std::vector<bool> *closedFlags = nullptr);
    std::vector<std::vector<Eigen::Vector2i>> findConnectedEdgeSegments(const std::vector<Eigen::Vector2i> &edges);

    std::vector<Eigen::Vector2f> intersectLinesLine2D(const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> &ons, const Eigen::Vector2f &o, const Eigen::Vector2f &n);
    std::pair<Eigen::Vector2f, float> computePolygonCentroid2D(const std::vector<Eigen::Vector2f> &pts, bool withArea);

    std::vector<std::vector<Eigen::VectorXf>> concatPolyParts(const std::vector<std::vector<Eigen::VectorXf>> &polyParts);
    std::vector<std::vector<Eigen::Vector2f>> concatPolyParts(const std::vector<std::vector<Eigen::Vector2f>> &polyParts);
    std::vector<Eigen::VectorXf> limitedDissolve2D(const std::vector<Eigen::VectorXf> &verts);
    Eigen::MatrixXf randomJitter(int count, int dim, float scale);
    Eigen::MatrixXf generateGridPoints(int n, int d, float e = 1.f);
    Eigen::MatrixXf generateJitteredGridPoints(int n, int d, float e = 1.f);
    std::vector<std::pair<int, int>> computeFaceCutIdxs(const std::vector<std::vector<int>> &faceMasks);
    bool haveCommonElement(
        const std::vector<int> &a,
        const std::vector<int> &b);

    inline float distPointToPlane(
        const Eigen::VectorXf &p,
        const Eigen::VectorXf &o,
        const Eigen::VectorXf &n)
    {
        return std::abs((p - o).dot(n));
    }

    bool vecsParallel(
        const Eigen::VectorXf &u,
        const Eigen::VectorXf &v,
        float eps,
        bool signedParallel = false);

    bool planesEquiv(
        const std::pair<Eigen::VectorXf, Eigen::VectorXf> &onA,
        const std::pair<Eigen::VectorXf, Eigen::VectorXf> &onB,
        float eps);

    Eigen::VectorXf inner1d(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B);
    // 法线归一化，类似 numpy normVec
    Eigen::MatrixXf normVec(const Eigen::MatrixXf &vecs);

    std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> computeCutPlanesVectorized(
        const Eigen::MatrixXf &sitesA,
        const Eigen::MatrixXf &sitesB,
        const Eigen::MatrixXf &vecsA,
        const Eigen::MatrixXf &vecsB,
        const Eigen::VectorXf &lambdasA,
        const Eigen::VectorXf &lambdasB);
}