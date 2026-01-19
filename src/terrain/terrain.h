#pragma once
#include <raylib.h>
#include <vector>
#include <cstdint>
#include <Eigen/Core>
#include "render.h"
#include "renderUtil.h"
#include "geo.h"
#include <queue>
#include "OpenSimplexNoise.h"
#include "tensorField.h"

namespace terrain
{

    using geo::Circle;
    using OpenSimplexNoise::Noise;

    struct TerrainVertex
    {
        Eigen::Vector3f position;
        Eigen::Vector3f normal;
        float slope;
        float aspect;
    };

    struct TerrainMeshData
    {
        std::vector<TerrainVertex> vertices;
        std::vector<uint32_t> indices;
        int gridWidth = 0;
        int gridHeight = 0;
    };

    struct TerrainFaceInfo
    {
        Eigen::Vector3f normal;
        float slope;
        float aspect;
    };

    struct ContourLayer
    {
        float height;
        std::vector<geo::Segment> segments;
    };

    struct TerrainCell
    {
        int v0, v1, v2, v3;
        int tri0[3];
        int tri1[3];
    };

    enum class TerrainViewMode
    {
        Lit,
        Wire,
        Aspect,
        Slope,
        Score
    };

    struct ContourShowData
    {
        bool isShowContours = false;
        //-----------------line-----------------
        bool isShowLines = false;
        Color lineColor = RL_BLACK;
        float lineColorF[4];
        //-----------------pt-------------------
        bool isShowPts = false;
        Color ptColor = RL_RED;
        float ptColorF[4];
        float ptSize = 0.1f;

        //-----------------pt_indices-------------
        bool isShowIndex = false;
        Color ptIndexColor = RL_DARKBLUE;
        float ptIndexColorF[4];
        float ptIndexSize = 18.f;
        ContourShowData()
        {
            syncFloatFromColor();
        }

        void syncFloatFromColor()
        {
            auto toFloat4 = [](Color c, float f[4])
            {
                f[0] = c.r / 255.0f;
                f[1] = c.g / 255.0f;
                f[2] = c.b / 255.0f;
                f[3] = c.a / 255.0f;
            };

            toFloat4(lineColor, lineColorF);
            toFloat4(ptColor, ptColorF);
            toFloat4(ptIndexColor, ptIndexColorF);
        }
    };

    struct RegionGrowConfig
    {
        float radius = 10.f;        // R
        int targetCount = 200;      // C
        int maxBadCount = 20;       // 允许的 score<0 数量
        float badRatioLimit = 0.2f; // 允许坏点比例
        float maxRadius = 20;
    };

    struct ChunkMeshInfo
    {
        Mesh mesh;                          // raylib mesh
        std::vector<int> localToGlobalVert; // local vertexid -> global vertexid
        std::vector<int> localToGlobalFace;
    };

    struct RegionInfo
    {
        std::vector<int> faces;
        int centerFace = -1;
        int centeridx = -1;
    };

    struct Road
    {
        std::vector<int> path;
        int level;
    };

    struct RoadSeed
    {
        int startVertex;
        Eigen::Vector3f forward;
        int level;
    };

    struct ParcelFaceNode
    {
        int id = -1;
        int face = -1;
        float dist = static_cast<float>(1e9);
        Eigen::Vector2f dirToRoad;
        bool processed = false;
        ParcelFaceNode() = default;
    };


    struct FacesParcel{
        int parcelID = -1;
        std::vector<int> faces;
        FacesParcel() = default;
    };

    class Terrain
    {
    public:
        // terrain generation
        int octaves = 4;
        float lacunarity = 1.6f;
        float gain = 0.7f;

        // terrain path cost weight
        float w_slope = 53.5f;
        float w_dist = 10.5f;
        float w_decent = 2.f;
        float w_up = 24.5f;
        float w_down = 14.5;

        // terrain vertex sorce weight
        float wv_slope = 10.f;
        float wv_aspect = 5.f;
        float score_threshold = 0.62f;
        int minRegionFaceSize = 200;
        RegionGrowConfig regionConfig;
        std::vector<RegionInfo> regionInfos;
        bool additionalShowWire = false;

        Terrain(int width, int height, float cellSize);
        Terrain(int seed_, int width, int height, float cellSize, float frequency, float amplitude);
        void regenerate(int w, int h, float freq, float amp);
        void regenerate(int seed, int w, int h, float freq, float amp);
        void generateHeight(float frequency, float amplitude);
        void buildMesh();
        void calculateInfos();
        // 上传到 GPU
        void upload();
        void draw() const;

        // member variables function
        const TerrainMeshData &getMesh() const { return mesh; }
        const std::vector<Vector3> getMeshVertices();
        const std::vector<TerrainCell> &getCells() const { return cells; }
        const std::vector<TerrainFaceInfo> &getFaceInfos() const { return faceInfos; }
        const TerrainViewMode &getViewMode() const { return viewMode; }
        const int &getWidth() const { return width; }
        const int &getHeight() const { return height; }
        const float &getCellSize() const { return cellSize; }
        const bool &getContousShow() const { return isShowContours; }
        const float &getMinHeight() const { return minHeight; }
        const float &getMaxHeight() const { return maxHeight; }
        const ContourShowData &getContourShowData() const { return contourShowData; }
        const std::vector<float> &getFaceScores() const { return scores; }
        const Eigen::AlignedBox2f getAABB2() const { return aabb2; }

        //----------------------utils--------------------
        inline int vertexIndex(int gx, int gy) const { return gy * (width + 1) + gx; }
        inline int gridIndex(int gx, int gy) const { return gy * width + gx; }

        void applyFaceColor();                                                                   // set face colors depend on slope , aspect
        bool sampleTensorAt(field::TerrainTensor<float> &out, const Eigen::Vector2f &pos) const; // sample vertex slope and aspect
        std::unordered_map<int, field::TerrainTensor<float>> sampleTensorAtGrids(const std::vector<Eigen::Vector2f> &grids) const;
        bool projectPolylineToTerrain(const std::vector<Eigen::Vector2f> &polyline2D, std::vector<Eigen::Vector3f> &outPolyline3D) const;
        void setViewMode(TerrainViewMode mode); // set viewmode to draw different analysis

        void setContoursShow(bool contourShow);
        std::vector<geo::Segment> extractContourAtHeight(float isoHeight) const;                              // extract contours at specific height
        std::vector<ContourLayer> extractContours(float gap) const;                                           // extract contours at each gap
        void drawContours(const std::vector<ContourLayer> &layers) const;                                     // draw contour infos
        void drawContourPtIndices(const std::vector<ContourLayer> &layers, render::Renderer3D &render) const; // draw contour pts indices
        void buildContourSettings();

        //----------------------path finding utils on terrain --------------------------
        inline bool valid(int x, int y) const { return x >= 0 && x <= width && y >= 0 && y <= height; }
        std::vector<geo::GraphEdge> buildGraph(int rank) const;
        void debugBuildGraphAt(int cx, int cy, int rank) const;
        void drawGraphEdges(int cx, int cy, int rank) const;
        float edgeCost(int a, int b) const;
        std::vector<std::vector<geo::GraphEdge>> buildAdjacencyGraph(int rank) const;
        std::vector<int> shortestPathDijkstra(int start, int goal, const std::vector<std::vector<geo::GraphEdge>> &adj) const;
        void drawPath(const std::vector<int> &path, float width) const;
        void computeRegionCenter(RegionInfo &region) const;
        Eigen::Vector3f computeForwardDirection(const std::vector<int> &path, int i) const;
        std::vector<std::vector<int>> buildMainRoads(std::vector<RegionInfo> &regions, int mainRegionCount, const std::vector<std::vector<geo::GraphEdge>> &adj) const;
        std::vector<int> sampleVerticesByDistance(const std::vector<int> &path, float interval) const;
        float pathLength(const std::vector<int> &path) const;

        std::vector<Road> buildRoads(std::vector<Eigen::Vector3f> &seedPoints, std::vector<Eigen::Vector3f> &controlPts, std::vector<RegionInfo> &regions, int mainRegionCount, const std::vector<std::vector<geo::GraphEdge>> &adj);
        std::vector<field::Polyline2_t<float>> convertRoadToFieldLine(const std::vector<Road> roads) const;
        void drawRoads(const std::vector<Road> &roads, float MaxWidth) const;

        //---------------------- evaluate score utils -----------------------------
        std::vector<float> evaluateVertexScore(const std::vector<float> &faceScores) const;
        std::vector<float> evaluateFaceScore() const;
        std::vector<std::vector<int>> buildFaceAdjacency() const;
        std::vector<std::vector<int>> floodFillFaces(const std::vector<float> &scores, float threshold) const;
        std::vector<std::vector<int>> floodFillFacesSoft(std::vector<float> &scores, float threshold) const;
        bool isRegionCompact(const std::vector<int> &region, const Eigen::Vector2f &center) const;

        //---------------------- tensorfield utils --------------------------------
        std::vector<FacesParcel> generateParcelsWithRoads(const field::TensorField2D<float>& field,const std::vector<Road> &roads,float parcelWidth,float parcelDepth) const;

        inline Eigen::Vector2f faceCenter(int f) const
        {
            int i0 = mesh.indices[f * 3 + 0];
            int i1 = mesh.indices[f * 3 + 1];
            int i2 = mesh.indices[f * 3 + 2];

            Eigen::Vector3f c =
                (mesh.vertices[i0].position +
                 mesh.vertices[i1].position +
                 mesh.vertices[i2].position) /
                3.0f;

            return {c.x(), c.y()};
        }

        inline Eigen::Vector3f faceCenter3D(int f) const
        {
            int i0 = mesh.indices[f * 3 + 0];
            int i1 = mesh.indices[f * 3 + 1];
            int i2 = mesh.indices[f * 3 + 2];

            Eigen::Vector3f c =
                (mesh.vertices[i0].position +
                 mesh.vertices[i1].position +
                 mesh.vertices[i2].position) /
                3.0f;

            return c;
        }

        inline bool insideCircle(int f, const Circle &c) const
        {
            Eigen::Vector2f p = faceCenter(f);
            return (p - c.center).norm() <= c.radius;
        }

        std::vector<int> computeSeedRegion(int seedFace, const std::vector<float> &scores) const;

    private:
        int seed;
        int width;      // grids number of x axis
        int height;     // grids number of x axis
        float cellSize; // grid edge length
        bool isShowContours = false;
        ContourShowData contourShowData; // contour data to draw
        float minHeight;                 // minimum height of terrain
        float maxHeight;                 // maximum height of terrain
        Eigen::AlignedBox2f aabb2;       // terrain planar aabb box
        std::vector<ChunkMeshInfo> chunkMeshes;
        std::vector<float> scores;
        std::vector<float> vertexScores;
        std::vector<std::vector<int>> regions;

        // mesh CPU data
        std::vector<TerrainFaceInfo> faceInfos; // face datas of normal , aspect ,slope
        std::vector<float> heightmap;           // reserve each vertex height from each row at x axis
        TerrainMeshData mesh;                   // z axis up direction mesh data
        std::vector<TerrainCell> cells;
        TerrainViewMode viewMode;  // show terrain analysis draw mode
        std::vector<Model> models; // raylib show data mesh
    };

    //-------------------------------------utils-----------------------------------------
    float noise2D(int x, int y);

    Mesh buildRaylibMesh(const TerrainMeshData &src);
    std::vector<ChunkMeshInfo> buildRaylibMeshes(const TerrainMeshData &src, int chunkSize);

    inline int igcd(int a, int b)
    {
        if (b == 0)
            return a;
        return igcd(b, a % b);
    }

    inline float fade(float t)
    {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    inline float lerp(float a, float b, float t)
    {
        return a + t * (b - a);
    }

    inline float grad(int hash, float x, float y)
    {
        int h = hash & 7; // 8 directions
        float u = h < 4 ? x : y;
        float v = h < 4 ? y : x;
        return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
    }

    inline Color transformFloat4ToColor(float colorf[])
    {
        return {
            (unsigned char)(colorf[0] * 255.0f),
            (unsigned char)(colorf[1] * 255.0f),
            (unsigned char)(colorf[2] * 255.0f),
            (unsigned char)(colorf[3] * 255.0f),
        };
    }

    inline float fbm2D_OpenSimplex(
        const Noise &noise,
        float x, float y,
        int octaves = 5,
        float lacunarity = 2.0f,
        float gain = 0.5f)
    {
        float sum = 0.0;
        float amp = 1.0;
        float freq = 1.0;

        for (int i = 0; i < octaves; ++i)
        {
            sum += (float)amp * noise.eval(x * freq, y * freq);
            freq *= lacunarity;
            amp *= gain;
        }
        return sum;
    }
}