#pragma once
#include <raylib.h>
#include <vector>
#include <cstdint>
#include <Eigen/Core>
#include "render.h"
#include "renderUtil.h"
#include "geo.h"
#include <queue>
namespace terrain
{

    struct TerrainVertex
    {
        Eigen::Vector3f position;
        Eigen::Vector3f normal;
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
        Slope
    };

    struct ContourShowData
    {
        bool isShowContours = false;
        //-----------------line-----------------
        bool isShowLines = false;
        Color lineColor = BLACK;
        float lineColorF[4];
        //-----------------pt-------------------
        bool isShowPts = false;
        Color ptColor = RED;
        float ptColorF[4];
        float ptSize = 0.1f;

        //-----------------pt_indices-------------
        bool isShowIndex = false;
        Color ptIndexColor = DARKBLUE;
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

    class Terrain
    {
    public:
        float w_slope = 3.f;
        float w_dist = 10.f;
        float w_decent = 2.f;
        float w_up = 0.2f;
        float w_down = 0.2f;
        
        Terrain(int width, int height, float cellSize);
        Terrain(int width, int height, float cellSize, float frequency, float amplitude);
        void generateHeight(float frequency, float amplitude);
        void buildMesh();
        // 上传到 GPU
        void upload();
        void draw() const;

        // member variables function
        const TerrainMeshData &getMesh() const { return mesh; }
        const std::vector<Vector3> getMeshVertices();
        const std::vector<TerrainCell> &getCells() const { return cells; }
        const std::vector<TerrainFaceInfo> &getFaceInfos() const { return faceInfos; }
        const int &getWidth() const { return width; }
        const int &getHeight() const { return height; }
        const bool &getContousShow() const { return isShowContours; }
        const float &getMinHeight() const { return minHeight; }
        const float &getMaxHeight() const { return maxHeight; }
        const ContourShowData &getContourShowData() const { return contourShowData; }

        //----------------------utils--------------------
        inline int vertexIndex(int gx, int gy) const
        {
            return gy * (width + 1) + gx;
        }

        void applyFaceColor(TerrainViewMode mode); // set face colors depend on slope , aspect
        void setViewMode(TerrainViewMode mode);    // set viewmode to draw different analysis
        void setContoursShow(bool contourShow);
        std::vector<geo::Segment> extractContourAtHeight(float isoHeight) const;                       // extract contours at specific height
        std::vector<ContourLayer> extractContours(float gap) const;                                    // extract contours at each gap
        void drawContours(std::vector<ContourLayer> layers) const;                                     // draw contour infos
        void drawContourPtIndices(std::vector<ContourLayer> layers, render::Renderer3D &render) const; // draw contour pts indices
        void buildContourSettings();

        //----------------------path finding utils on terrain --------------------------
        inline bool valid(int x, int y) const
        {
            return x >= 0 && x <= width && y >= 0 && y <= height;
        }

        void linkEdge(int a, int b, int rank, std::vector<geo::GraphEdge> &edges) const; //
        std::vector<geo::GraphEdge> buildGraph(int rank) const;
        void debugBuildGraphAt(int cx, int cy, int rank) const;
        void drawGraphEdges(int cx, int cy, int rank) const;
        float edgeCost(int a, int b) const;
        std::vector<int> shortestPath(int start,int goal,const std::vector<std::vector<int>> &adj) const;

        std::vector<std::vector<geo::GraphEdge>> buildAdjacencyGraph(int rank) const;
        std::vector<int> shortestPathDijkstra(int start, int goal,const std::vector<std::vector<geo::GraphEdge>> &adj) const;
        void drawPath( const std::vector<int> &path, float width) const;

    private:
        int width;      // grids number of x axis
        int height;     // grids number of x axis
        float cellSize; // grid edge length
        bool isShowContours = false;
        ContourShowData contourShowData; // contour data to draw
        float minHeight;                 // minimum height of terrain
        float maxHeight;                 // maximum height of terrain

        // mesh CPU data
        std::vector<TerrainFaceInfo> faceInfos; // face datas of normal , aspect ,slope
        std::vector<float> heightmap;           // reserve each vertex height from each row at x axis
        TerrainMeshData mesh;                   // z axis up direction mesh data
        std::vector<TerrainCell> cells;
        TerrainViewMode viewMode; // show terrain analysis draw mode
        Model model{};            // raylib show data mesh

        
    };

    //-------------------------------------utils-----------------------------------------
    float noise2D(int x, int y);

    Mesh buildRaylibMesh(const TerrainMeshData &src);

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

    // ----------------------------
    // 单 octave Perlin（连续）
    // ----------------------------
    inline float perlin2D(float x, float y)
    {
        static int perm[512];
        static bool initialized = false;

        if (!initialized)
        {
            int p[256] = {
                151, 160, 137, 91, 90, 15,
                131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
                140, 36, 103, 30, 69, 142, 8, 99, 37, 240,
                21, 10, 23, 190, 6, 148, 247, 120, 234, 75,
                0, 26, 197, 62, 94, 252, 219, 203, 117, 35,
                11, 32, 57, 177, 33, 88, 237, 149, 56, 87,
                174, 20, 125, 136, 171, 168, 68, 175, 74, 165,
                71, 134, 139, 48, 27, 166, 77, 146, 158, 231,
                83, 111, 229, 122, 60, 211, 133, 230, 220, 105,
                92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
                65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
                132, 187, 208, 89, 18, 169, 200, 196, 135, 130,
                116, 188, 159, 86, 164, 100, 109, 198, 173, 186,
                3, 64, 52, 217, 226, 250, 124, 123, 5, 202,
                38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
                59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
                223, 183, 170, 213, 119, 248, 152, 2, 44, 154,
                163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
                129, 22, 39, 253, 19, 98, 108, 110, 79, 113,
                224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
                251, 34, 242, 193, 238, 210, 144, 12, 191, 179,
                162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
                49, 192, 214, 31, 181, 199, 106, 157, 184, 84,
                204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
                138, 236, 205, 93, 222, 114, 67, 29, 24, 72,
                243, 141, 128, 195, 78, 66, 215, 61, 156, 180};
            for (int i = 0; i < 256; i++)
                perm[i] = perm[i + 256] = p[i];
            initialized = true;
        }

        int X = (int)std::floor(x) & 255;
        int Y = (int)std::floor(y) & 255;

        x -= std::floor(x);
        y -= std::floor(y);

        float u = fade(x);
        float v = fade(y);

        int aa = perm[perm[X] + Y];
        int ab = perm[perm[X] + Y + 1];
        int ba = perm[perm[X + 1] + Y];
        int bb = perm[perm[X + 1] + Y + 1];

        return lerp(
            lerp(grad(aa, x, y), grad(ba, x - 1, y), u),
            lerp(grad(ab, x, y - 1), grad(bb, x - 1, y - 1), u),
            v);
    }

    inline float fbm2D(float x, float y, int octaves = 5, float lacunarity = 2.0f, float gain = 0.5f)
    {
        float sum = 0.0f;
        float amp = 1.0f;
        float freq = 1.0f;

        for (int i = 0; i < octaves; ++i)
        {
            sum += amp * perlin2D(x * freq, y * freq);
            freq *= lacunarity;
            amp *= gain;
        }
        return sum;
    }
}