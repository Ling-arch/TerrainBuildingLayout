
#pragma once

#include <Eigen/Core>
#include <vector>
#include <functional>
#include "tensorField.h"
#include "terrain.h"
#include "render.h"

namespace SCARoad
{

    struct HalfEdge
    {
        int from;
        int to;
        float angle;
    };

    struct AttractorSettings
    {
        float attractionDistance = 50.0f;
        float killDistance = 10.0f;
    };
    struct Attractor
    {
        Eigen::Vector2f position;          // 吸引子位置
        std::vector<int> influencingNodes; // 当前帧影响的 node 索引
        bool fresh = true;                 // 是否是第一帧（用于 closed venation）
        AttractorSettings settings;
        bool reached = false;
        // 构造函数
        Attractor() = default;
        Attractor(const Eigen::Vector2f &pos, const AttractorSettings &settings)
            : position(pos), settings(settings) {}
        void resetFrame()
        {
            influencingNodes.clear();
        }
    };

    struct SCANode
    {
        Eigen::Vector2f position;
        int parent = -1;
        bool isTip = true;
        // 被哪些 attractor 影响（存 index）
        std::vector<int> influencedBy;
        std::vector<int> paths;
        std::vector<int> links;
        SCANode() = default;
    };

    class SCANetwork
    {
    public:
        std::vector<SCANode> nodes;
        std::vector<Attractor> attractors;
        using PointCloud2D = field::PointCloud2D<float>;
        float Litten_M_PI = field::Litten_M_PI;
        std::unique_ptr<PointCloud2D> cloud;
        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud2D>, PointCloud2D, 2>;
        std::unique_ptr<KDTree> tree;
        std::vector<Eigen::Vector2f> nodePoints;
        std::vector<std::pair<int, int>> edges; // 额外连接（闭环）
        std::vector<std::vector<int>> paths;    // 每条 path 是 node index 序列
        std::vector<int> nodeToPath;            // node -> pathID
        using TensorField = field::TensorField2D<float>;
        TensorField field;
        using Terrain = terrain::Terrain;
        Terrain terrain;
        float attractionDist = 60.f;
        float killDist = 3.f;
        

        enum class VenationType
        {
            Open,
            Closed
        } venationType = VenationType::Closed;

        SCANetwork();
        SCANetwork(const std::vector<SCANode> &nodesIn, const std::vector<Attractor> &attractIn, const Terrain &terrainIn, const TensorField &fieldIn)
            : nodes(nodesIn), attractors(attractIn),
              terrain(terrainIn), field(fieldIn)
        {
            initPaths();
            buildKDTree();
        }



        void initPaths();
        void buildKDTree();
        void rebuildNet(const std::vector<SCANode> &nodesIn,const std::vector<Attractor> &attractorsIn);
        std::vector<int> getNodesInRadius(const Eigen::Vector2f &pos, float radius) const;
        int getClosestNode(int attractorID);
        std::vector<int> getRelativeNeighbors(int attractorID) const;
        std::vector<int> getRelativeNeighbors(const Attractor &a) const;
        std::vector<int> getRelativeNeighbors(Eigen::Vector2f pos, float radius) const;
        Eigen::Vector2f getAverageDirection(int nodeID) const;
        bool tooCloseInSamePath(int a, int b, int minStep) const;
        int findNearestBranchDist(int pathID, int nodeID) const;
        std::vector<SCANode> growNodes(float stepSize, int minBranchGap);
        bool isAttractorKilled(const Attractor &a) const;
        void loopConnect(int forbiddenGap, int CONNECT_DIST, int MIN_BRANCH_GAP);
        void finalConnectNodes(int forbiddenGap, int CONNECT_DIST, int MIN_BRANCH_GAP);
        void update(bool &growthStopped);
        Eigen::Vector2f getGuidedDirection(int nodeID, const Eigen::Vector2f &candidateDir, float blend = 0.3f) const;
        Eigen::Vector2f getConstraintDirection(int nodeID, const Eigen::Vector2f &dir) const;
        void replacePathID(int nodeID, int oldPID, int newPID);
        std::vector<std::vector<Eigen::Vector2f>> extractRoads() const;
        using Polyline2_t = geo::Polyline2_t<float>;
        std::pair<std::vector<Polyline2_t>, std::vector<Polyline2_t>> extractCloseAndLinearRoads() const;
        std::vector<std::pair<int, std::unordered_set<int>>> collectEndpointWithForbidden(int forbiddenGap) const;
        std::unordered_set<int> filterAllPathEndpoints(int minLen) const;
        void drawNodesWithIndices(render::Renderer3D render) const;
        void debugPrintRelativeNeighbors(int nodeIndex, float radius) const;
        void debugPrintNodePaths(int nodeIndex) const;
        void debugPrintNodeLinks(int nodeIndex) const;
        void debugConnectNodes(int nodeIndex, float CONNECT_DIST) const;
    };

    inline float randomFloat(float min, float max)
    {
        return min + (max - min) * (float(rand()) / RAND_MAX);
    }

    inline float mapValue(float v, float inMin, float inMax, float outMin, float outMax)
    {
        float t = (v - inMin) / (inMax - inMin);
        return outMin + t * (outMax - outMin);
    }

    std::vector<Attractor> getRandomAttractors(float gap, float width, float height, std::vector<Eigen::Vector2f> &pos, const Eigen::Vector2f &center = Eigen::Vector2f(0.0f, 0.0f));
    geo::Polyline2_t<float> buildPolyline(
        const std::vector<SCANode> &nodes,
        const std::vector<int> &path);
}
