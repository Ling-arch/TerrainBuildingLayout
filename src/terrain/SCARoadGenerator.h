
#pragma once

#include <Eigen/Core>
#include <vector>
#include <functional>
#include "tensorField.h"

namespace SCARoad
{
    struct AttractorSettings
    {
        float attractionDistance = 50.0f;
        float killDistance = 10.0f;
    };
    struct Attractor
    {

        // 数据
        Eigen::Vector2f position;          // 吸引子位置
        std::vector<int> influencingNodes; // 当前帧影响的 node 索引
        bool fresh = true;                 // 是否是第一帧（用于 closed venation）
        AttractorSettings settings;
        bool reached = false;
        // 构造函数
        Attractor() = default;

        Attractor(const Eigen::Vector2f &pos, const AttractorSettings &settings)
            : position(pos), fresh(true), settings(settings) {}

        // 每帧清理（很重要）
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
        std::vector<int> extraLinks;
        SCANode() = default;
    };

    class SCANetwork
    {
    public:
        std::vector<SCANode> nodes;
        std::vector<Attractor> attractors;

        // KDTree
        using PointCloud2D = field::PointCloud2D<float>;
        float Litten_M_PI = field::Litten_M_PI;
        std::unique_ptr<PointCloud2D> cloud;
        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud2D>, PointCloud2D, 2>;
        std::unique_ptr<KDTree> tree;
        std::vector<Eigen::Vector2f> nodePoints;
        std::vector<std::pair<int, int>> edges; // 额外连接（闭环）
        float attractionDist = 50.f;
        float killDist = 10.f;
        float minBranchAngle = 20.0f;
        float maxBranchAngle = 90.0f;
        int internodeLength = 5;

        // ===== Phase 2 =====
        float connectDist = 8.0f;
        int maxExtraLinks = 2;
        float idealAngleWeight = 0.7f;

        enum class VenationType
        {
            Open,
            Closed
        } venationType = VenationType::Closed;
        void buildKDTree();
        std::vector<int> getNodesInRadius(const Eigen::Vector2f &pos, float radius) const;
        int getClosestNode(int attractorID);
        std::vector<int> getRelativeNeighbors(int attractorID) const;
        Eigen::Vector2f getAverageDirection(int nodeID) const;
        void update(bool &growthStopped);
        std::vector<std::vector<Eigen::Vector2f>> extractRoads() const;
        Eigen::Vector2f clampDirection(
            const Eigen::Vector2f &parentDir,
            const Eigen::Vector2f &newDir) const;
        float computeIdealAngleScore(int nodeID, int candidateID) const;
        bool formsTriangle(int a, int b) const;
        void buildConnections();
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

    std::vector<Attractor> getRandomAttractors(int num, float width, float height, std::vector<Eigen::Vector2f> &pos, const Eigen::Vector2f &center = Eigen::Vector2f(0.0f, 0.0f));

}
