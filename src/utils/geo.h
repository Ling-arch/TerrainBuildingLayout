#pragma once
#include <Eigen/Core>
#include "polyloop.h"
#include <raylib.h>


namespace geo
{
    struct Segment
    {
        Eigen::Vector3f p0;
        Eigen::Vector3f p1;
    };

    struct PointKey
    {
        int x, y, z;
        bool operator==(const PointKey &o) const
        {
            return x == o.x && y == o.y && z == o.z;
        }
    };

    PointKey makeKey(const Eigen::Vector3f &p, float eps);

    struct PointKeyHash
    {
        size_t operator()(const PointKey &k) const
        {
            return ((size_t)k.x * 73856093u) ^
                   ((size_t)k.y * 19349663u) ^
                   ((size_t)k.z * 83492791u);
        }
    };

    struct Polyline
    {
        std::vector<Eigen::Vector3f> points;
        bool closed = false;
    };

    struct Edge
    {
        int segId;
        int end; // 0 = p0, 1 = p1
    };

    struct GraphEdge
    {
        int from;
        int to;
        float cost;
        int dx;
        int dy;
    };

    struct Circle
    {
        Eigen::Vector2f center; // face 质心
        float radius;
    };

    std::vector<Polyline> buildPolylines(const std::vector<geo::Segment> &segments, float eps = 1e-4f);

    void followChain(
        int startSeg,
        int startEnd,
        const std::vector<geo::Segment> &segments,
        const std::unordered_map<PointKey, std::vector<Edge>, PointKeyHash> &adj,
        std::vector<bool> &used,
        float eps,
        Polyline &out);

    struct Vertex
    {
        Eigen::Vector3f position;
        Eigen::Vector3f normal;
    };

    struct MeshData
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices; // triangle list
    };


    class PolygonMesh
    {
    public:
        MeshData mesh;
        PolygonMesh() = default;
        PolygonMesh(const std::vector<Eigen::Vector3f> &points_, float height_)
        : points(points_), height(height_)  
        {
            build();
            upload();

        }
        void build();
        void upload();
        void regenerate(float newHeight);
        void draw(Color color, float colorAlpha, bool outline, bool wireframe,  float wireframeAlpha);

    private:
        std::vector<Eigen::Vector3f> points;
        float height;
        Model model;
    };

    Mesh buildRaylibMesh(const MeshData &src);
}