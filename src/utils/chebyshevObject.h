#pragma once

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
#include <cassert>
#include <functional>
#include <string>
#include <map>
#include <memory>
#include "geo.h"
#include "chebyshevUtils.h"
#include "render.h"
#include "renderUtil.h"


#define CHEBYSHEV_DEBUG

#ifdef CHEBYSHEV_DEBUG
#define DBG(msg) std::cout << msg << std::endl
#else
#define DBG(msg)
#endif

namespace infinityVoronoi
{
    using Eigen::MatrixXf;
    using Eigen::MatrixXi;
    using Vec = Eigen::VectorXf;
    using Vec2 = Eigen::Vector2f;
    using Vec3 = Eigen::Vector3f;
    using Vec2i = Eigen::Vector2i;

    bool sectorsDoIntersect(const std::vector<MatrixXf> &sGeometry);

    inline MatrixXf vecArrayToMatrix(const std::vector<Vec> &verts)
    {
        if (verts.empty())
            return MatrixXf();

        int dim = verts[0].size();

        MatrixXf M(verts.size(), dim);

        for (int i = 0; i < (int)verts.size(); ++i)
            M.row(i) = verts[i].transpose();

        return M;
    }

    struct Plane2
    {
        Vec2 o;
        Vec2 n;
    };

    class SectorCutObject
    {
    public:
        std::vector<int> hullPlaneKeys;
        std::map<int, std::vector<int>> polys;

    public:
        SectorCutObject() = default;
        virtual ~SectorCutObject() = default;
        virtual void cutWithPlane(const Vec2 &o, const Vec2 &n, int cutPlaneKey) = 0;
        virtual void computePolysCentroidsAndWeights() = 0;
        virtual std::vector<Vec> getPolysCentroids(bool ioClipped = true) = 0;
        virtual std::vector<float> getPolysWeights(bool ioClipped = true) = 0;
        virtual void setPolyIoLabels(const std::vector<bool> &msk) = 0;
        virtual std::vector<std::vector<Vec>> getHullVerts() = 0;
        virtual std::vector<Vec> getVertices() = 0;
        //virtual std::vector<int> getPolyKeys() const = 0;
        virtual std::vector<Vec2> buildPolyFromKey(int k) const = 0;
    };

    struct PyraCutObject : public SectorCutObject
    {
    };

    struct TriCutObject : public SectorCutObject
    {
        // ---------- static ----------
        std::vector<Vec2> initCellVerts;

        // ---------- topology ----------
        std::vector<Vec2i> edges;
        //std::map<int, std::vector<int>> polys;
        std::vector<Vec2i> edgePolyIdxs;

        // ---------- geometry ----------
        std::vector<Vec2> vertices;
        std::unordered_map<int, Plane2> edgesPlanes;
        std::vector<int> edgePlaneKeys;

        // ---------- poly info ----------
        std::vector<Vec2> polysCentroids;
        std::vector<float> polysAreas;
        std::map<int, bool> polysIoLabel;
        std::vector<int> cellPolyIdxs;

        // ---------- hull ----------
        // std::vector<int> hullPlaneKeys;

        // ---------- ctor ----------
        TriCutObject(
            const Vec2 &site,
            int di,
            const std::vector<float> &scale,
            const Eigen::Matrix2f &M);

        // ---------- core ops ----------
        void clipWithPlane(const Vec2 &o, const Vec2 &n, int cutPlaneKey);
        void cutWithPlane(const Vec2 &o, const Vec2 &n, int cutPlaneKey) override;

        // ---------- poly queries ----------
        void computePolysCentroidsAndWeights() override;
        std::vector<Vec> getPolysCentroids(bool ioClipped = true) override;
        std::vector<float> getPolysWeights(bool ioClipped = true) override;
        std::vector<Vec2> buildPolyFromKey(int key) const;
        // ---------- hull ----------
        std::vector<std::vector<Vec>> getHullVerts() override;
        std::vector<Vec> getVertices() override;
        // ---------- io mask ----------
        void setPolyIoLabels(const std::vector<bool> &msk) override;
    };

    void clipCellGeometry(
        SectorCutObject &cellSec,
        const Vec &siteA,
        const Vec &MvecA,
        float lambdaA,
        const std::vector<Vec> &sitesB,
        const std::vector<Eigen::MatrixXf> &MvecsB,
        const std::vector<Vec> &lambdasB,
        float domainExtent);

    struct DissolveResult
    {
        std::vector<std::vector<Vec>> polys;
        std::vector<long long> planeKeys;
    };

    DissolveResult dissolve(
        const std::vector<std::unique_ptr<SectorCutObject>> &sectors,
        int nDim);

    
    class ChebyshevObject
    {
    public:
        // orientation function
        using OriFun = std::function<Vec(const MatrixXf &)>;
        // Anisotropy function
        using AniFun = std::function<MatrixXf(const MatrixXf &)>;

        ChebyshevObject(
            const MatrixXf &sites,
            OriFun oriFun = nullptr,
            AniFun aniFun = nullptr,
            float extent = 1.0f,
            int nDim = 2);

        virtual ~ChebyshevObject() = default;

        void computeDiagram(bool finish = true);
        void lloydRelax(float itersThresh = 0.0f);

        const std::vector<std::vector<std::vector<Vec>>> &getCellVertexSet() const { return cellVertexSets; }
        const int &getNumSites() const { return numSites; }
        void drawSiteWithDirs(Color c, float thickness = 0.03f, float scale = 1.f) const;
        void drawCutPlanes(Color c,float thickness = 0.03f,float extendScale = 1.f) const;
        void drawCellSectors(float thickness = 0.02f) const;
        void drawCellSector(
            int sIdx,
            int di,
            float thickness = 0.02f) const;
        void debugPrintSitesAndLambdas() const;

    public:
        /* ---------- abstract parts ---------- */
        virtual void
        logMeta() = 0;

        virtual std::vector<MatrixXf> getSitesSectorGeometry(int sIdx, int di, int sJdx, int dj) = 0;

        virtual void processDissolvedSectors(const std::vector<DissolveResult> &dissolved) = 0;

        /* ---------- pipeline ---------- */
        void initCells();
        void computeNeighborsAndPlanes();
        void computeCutPlane(int sA, int dA,
                             int sB, int dB,
                             Vec &pO,
                             Vec &pN);
        void addCutPlane(int sIdx, int di, int sJdx, int dj, const Vec &o, const Vec &n);
        void cutWithPlanes();
        void clipCellGeometry();
        void dissolveCells();
        void finish();

    protected:
        /* ---------- config ---------- */
        int nDim;
        int numSites;
        float domainExtent;
        MatrixXf baseDir;
        /* ---------- site data ---------- */
        MatrixXf sites;
        std::vector<int> sIdxs;
        float cellScale = 1.f;
        std::vector<float> cellScales;
        OriFun oriFun;
        AniFun aniFun;

        /* ---------- metric ---------- */
        std::vector<MatrixXf> Ms;    // nDim x nDim
        std::vector<MatrixXf> Mvecs; // (2*nDim) x nDim
        MatrixXf lambdas;            // numSites x (2*nDim)

        /* ---------- domain ---------- */
        std::vector<MatrixXf> domainPlanes;
        MatrixXf boxVerts;

        /* ---------- topology ---------- */
        std::vector<std::vector<std::vector<std::pair<int, int>>>> sitesNeighbors;

        // cutPlaneKey -> ((o,n),(sIdx,sJdx))
        std::map<long long,
                 std::pair<std::pair<Vec, Vec>, std::pair<int, int>>>
            cutPlanes;

        std::vector<std::vector<std::vector<long long>>> cutPlaneKeys;
        std::vector<std::vector<std::vector<Vec>>> cellVertexSets;
        std::vector<std::vector<long long>> cellPlaneKeys;
        /* ---------- geometry ---------- */
        std::vector<std::vector<std::unique_ptr<SectorCutObject>>> cellSectors;

        /* ---------- results ---------- */
        MatrixXf cellCentroids;        // numSites × nDim
        MatrixXf cellBBcenters;        // numSites × nDim
        std::vector<MatrixXf> cellBBs; // 每个: (2^nDim × nDim)
        std::vector<std::vector<int>> cellAdjacency;
        std::vector<Vec2i> cellAdjacencyEdges;

        /* ---------- timing ---------- */
        std::vector<std::vector<float>> timings;
    };

    class ChebyshevObject2D : public ChebyshevObject
    {
    public:
        ChebyshevObject2D(
            const MatrixXf &sites,
            OriFun oriFun = nullptr,
            AniFun aniFun = nullptr,
            float extent = 1.f);

    protected:
        /* ---------- overrides ---------- */
        void logMeta() override;

        std::vector<MatrixXf>
        getSitesSectorGeometry(
            int sIdx, int di,
            int sJdx, int dj) override;

        void processDissolvedSectors(
            const std::vector<DissolveResult> &dData) override;
    };

    inline std::string vec2ToStr(const Vec2 &v)
    {
        std::ostringstream oss;
        oss << "(" << v.x() << ", " << v.y() << ")";
        return oss.str();
    }
}