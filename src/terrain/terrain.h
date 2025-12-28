#pragma once
#include <raylib.h>
#include <vector>
#include <cstdint>
#include <Eigen/Core>
#include "render.h"
#include "renderUtil.h"
#include "geo.h"

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

    class Terrain
    {
    public:
        Terrain(int width, int height, float cellSize);
        Terrain(int width, int height, float cellSize, float frequency, float amplitude);
        void generateHeight(float frequency, float amplitude);
        void buildMesh();
        // 上传到 GPU
        void upload();
        void draw() const;

        const TerrainMeshData &getMesh() const { return mesh; }
        const std::vector<Vector3> getMeshVertices();
        const std::vector<TerrainCell> &getCells() const { return cells; }
        const std::vector<TerrainFaceInfo> &getFaceInfos() const { return faceInfos; }
        const bool &getContousShow() const { return isShowContours; }
        const float &getMinHeight()const {return minHeight;}
        const float &getMaxHeight() const { return maxHeight; }
        int vertexIndex(int gx, int gy) const;

        void applyFaceColor(TerrainViewMode mode);
        void setViewMode(TerrainViewMode mode);
        void setContoursShow(bool contourShow);
        //----------------------utils--------------------
        std::vector<geo::Segment> extractContourAtHeight(float isoHeight) const;
        std::vector<ContourLayer>extractContours(float gap) const;
        void drawContours() const;

    private:
        int width;
        int height;
        float cellSize;
        bool isShowContours = false;
        float minHeight;
        float maxHeight;
        // CPU 数据
        std::vector<TerrainFaceInfo> faceInfos;
        std::vector<float> heightmap;
        TerrainMeshData mesh;
        std::vector<TerrainCell> cells;
        TerrainViewMode viewMode;
        Model model{};
    };

    //-------------------------------------utils-----------------------------------------
    float noise2D(int x, int y);

    Mesh buildRaylibMesh(const TerrainMeshData &src);

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