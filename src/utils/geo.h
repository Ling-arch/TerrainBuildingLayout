#pragma once
#include <Eigen/Core>


namespace geo{
    struct Segment{
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


    struct GraphEdge{
        int from;
        int to;
        float cost;
        int dx;
        int dy;
    };

    struct Circle{
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

   
}