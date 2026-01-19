#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <imgui.h>
#include <rlImGui.h>
#include <iostream>
#include <Eigen/SparseQR>
#include <nanoflann.hpp>
#include "render.h"
#include "geo.h"
#include "util.h"

namespace field
{
    constexpr double M_PI = 3.14159265358979323846;

    template <typename Scalar>
    using Vector2 = Eigen::Matrix<Scalar, 2, 1>;

    template <typename Scalar>
    using AABB = Eigen::AlignedBox<Scalar, 2>;

    template <typename Scalar>
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;

    template <typename Scalar>
    using Polyline2_t = geo::Polyline2_t<Scalar>;

    template <typename Scalar>
    struct Tensor
    {
        Vector2<Scalar> pos;
        Scalar theta;
        std::vector<Vector2<Scalar>> dirs;
        Tensor() = default;
        Tensor(Vector2<Scalar> p, Scalar theta_) : pos(p), theta(theta_) { buildFourDirs(); };
        void buildFourDirs()
        {
            dirs.clear();
            dirs.resize(4);
            dirs[0] = Vector2<Scalar>(std::cos(theta), std::sin(theta));
            dirs[1] = Vector2<Scalar>(-std::sin(theta), std::cos(theta));
            dirs[2] = Vector2<Scalar>(-std::cos(theta), -std::sin(theta));
            dirs[3] = Vector2<Scalar>(std::sin(theta), -std::cos(theta));
        }
    };

    template <typename Scalar>
    struct GridTensor
    {
        Vector2<Scalar> center;
        Scalar theta[4]; // 四个角点 theta
    };

    template <typename Scalar>
    struct PointCloud2D
    {
        const std::vector<Vector2<Scalar>> &pts;
        PointCloud2D(const std::vector<Vector2<Scalar>> &points) : pts(points) {}

        inline size_t kdtree_get_point_count() const { return pts.size(); }

        inline Scalar kdtree_get_pt(const size_t idx, int dim) const
        {
            return pts[idx][dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX &) const { return false; }
    };
    /*
     * Wrap angle theta to [0, pi/2)
     */
    template <typename Scalar>
    inline Scalar wrapTheta(Scalar theta)
    {
        const Scalar half_pi = Scalar(M_PI * 0.5);
        theta = std::fmod(theta, half_pi);
        if (theta < Scalar(0))
            theta += half_pi;
        return theta;
    }

    /*
     * Convert a 2D direction vector to an angle in [0, pi/2)
     */
    template <typename Scalar>
    inline Scalar directionToTheta(const Vector2<Scalar> &dir)
    {
        Vector2<Scalar> d = dir.normalized();
        Scalar theta = std::atan2(d.y(), d.x());
        return wrapTheta<Scalar>(theta);
    }

    template <typename Scalar>
    struct PointAtrractor
    {
        Vector2<Scalar> pos;
        Scalar radius;
        PointAtrractor() = default;
        PointAtrractor(const Vector2<Scalar> &pos_, Scalar radius_)
            : pos(pos_), radius(radius_) {}

        void draw() const
        {

            DrawCylinder({pos.x(), 0.f, -pos.y()}, radius, radius, 10.f, 20, Fade(RL_RED, 0.06f));
        }
    };

    template <typename Scalar>
    struct TerrainTensor
    {
        Vector2<Scalar> position;
        Scalar slope;
        Scalar aspect;
        int16_t init = -1;
        TerrainTensor() = default;
        TerrainTensor(const Vector2<Scalar> &pos, Scalar s, Scalar a)
            : position(pos), slope(s), aspect(a) {}
    };

    template <typename Scalar>
    class TensorField2D
    {
    public:
        TensorField2D() = default;
        TensorField2D(const AABB<Scalar> &bound, int minGridNum, const std::vector<Polyline2_t<Scalar>> &polylines);
        TensorField2D(const AABB<Scalar> &bound, int minGridNum, const std::vector<Polyline2_t<Scalar>> &polylines, const std::vector<PointAtrractor<Scalar>> &attractors);
        TensorField2D(const AABB<Scalar> &bound, int minGridNum);

        const AABB<Scalar> &getBound() const { return bound_; }
        const int &getMinGridNum() const { return minGridNum_; }
        const Scalar &getTerrainWeight() const { return terrainWeight; }
        const Scalar &getGridSize() const { return gridSize_; }
        const int &getGridNX() const { return nx_; }
        const int &getGridNY() const { return ny_; }
        const std::vector<Vector2<Scalar>> &getGridPoints() const { return gridPoints_; }
        const std::vector<Vector2<Scalar>> &getAllPoints() const { return allPoints_; }
        const std::vector<Tensor<Scalar>> &getTensors() const { return gridTensors_; }
        const std::vector<Tensor<Scalar>> &getAllTensors() const { return allTensors_; }
        const std::vector<Vector2<Scalar>> &getCellCenters() const { return cellCenters_; }
        std::vector<PointAtrractor<Scalar>> &getAttractorsRef() { return attractors_; }
        const std::vector<PointAtrractor<Scalar>> &getAttractors() const { return attractors_; }
        const std::vector<Polyline2_t<Scalar>> &getPolylines() const { return polylines_; }

        //-----------------------------functions-------------------------
        std::array<Vector2<Scalar>, 4> getTensorAt(const Vector2<Scalar> &pos) const;
        void addConstraint(const std::vector<Polyline2_t<Scalar>> &polylines, const std::vector<PointAtrractor<Scalar>> &attractors, const std::unordered_map<int, TerrainTensor<Scalar>> &terrainTensors);
        void setTensorWeight(Scalar weight);
        void resolveTensor();
        Tensor<Scalar> testTensorAt(const Vector2<Scalar> &pos) const;
        Polyline2_t<Scalar> traceStreamlineFromPos(const Vector2<Scalar> &pos, int uv, int maxIter = 1000) const;
        std::vector<Polyline2_t<Scalar>> genStreamlines(const std::vector<Vector2<Scalar>> &points, int maxIter = 1000) const;
        Polyline2_t<Scalar> traceStreamlineFromPosWithDir(const Vector2<Scalar> &pos, const Vector2<Scalar> &startDir, Scalar step = 1.5f, int maxIter = 20) const;
        std::unordered_map<int, std::vector<Polyline2_t<Scalar>>> traceCrossLinesBetweenGaps(int seed, int dir, Scalar rebuildStep, Scalar traceStep, const Eigen::Vector2i &gapRange) const;
        std::vector<Polyline2_t<Scalar>> trace2sidesParcels(int seed, Scalar rebuildStep, Scalar traceStep, const Eigen::Vector2i &gapRange/* , Scalar cellSize, int width, int height */) const;

    private:
        void buildGrid();
        void collectConstraintPoints();
        void buildConnectivity();
        void solveTensorField();
        void buildGridTensors();
        inline bool isInside(const Vector2<Scalar> &p) const { return bound_.contains(p); }

    private:
        // input
        AABB<Scalar> bound_;
        int minGridNum_;
        Scalar terrainWeight = Scalar(2);
        std::vector<Polyline2_t<Scalar>> polylines_;
        std::vector<PointAtrractor<Scalar>> attractors_;
        std::unordered_map<int, TerrainTensor<Scalar>> terrainTensors_;
        // grids
        Scalar gridSize_;
        int nx_, ny_;
        std::vector<Vector2<Scalar>> gridPoints_;
        // all points(grids + constrain points)
        std::vector<Vector2<Scalar>> allPoints_;
        std::vector<int> ptLabels;
        // constraints, vertex idx->0
        std::unordered_map<int, Scalar> fixedThethas_;

        // adjacency
        std::vector<std::pair<int, int>> edges_;

        // linear system
        Eigen::SparseMatrix<Scalar> A_;
        Eigen::VectorXd b_;
        Eigen::VectorXd theta_;

        // output
        std::vector<Tensor<Scalar>> gridPtTensors_;
        std::vector<Tensor<Scalar>> allTensors_;
        std::vector<GridTensor<Scalar>> gridTensors_;
        std::vector<Vector2<Scalar>> cellCenters_;
    };

    template <typename Scalar>
    TensorField2D<Scalar>::TensorField2D(const AABB<Scalar> &bound, int minGridNum, const std::vector<Polyline2_t<Scalar>> &polylines)
        : bound_(bound), minGridNum_(minGridNum), polylines_(polylines)
    {
        buildGrid();
        collectConstraintPoints();
        buildConnectivity();
        solveTensorField();
        buildGridTensors();
    }

    template <typename Scalar>
    TensorField2D<Scalar>::TensorField2D(const AABB<Scalar> &bound, int minGridNum, const std::vector<Polyline2_t<Scalar>> &polylines, const std::vector<PointAtrractor<Scalar>> &attractors)
        : bound_(bound), minGridNum_(minGridNum), polylines_(polylines), attractors_(attractors)
    {
        buildGrid();
        collectConstraintPoints();
        buildConnectivity();
        solveTensorField();
        buildGridTensors();
    }
    template <typename Scalar>
    TensorField2D<Scalar>::TensorField2D(const AABB<Scalar> &bound, int minGridNum)
        : bound_(bound), minGridNum_(minGridNum)
    {
        buildGrid();
    }

    template <typename Scalar>
    void TensorField2D<Scalar>::addConstraint(const std::vector<Polyline2_t<Scalar>> &polylines, const std::vector<PointAtrractor<Scalar>> &attractors, const std::unordered_map<int, TerrainTensor<Scalar>> &terrainTensors)
    {
        polylines_ = polylines;
        attractors_ = attractors;
        terrainTensors_ = terrainTensors;
        collectConstraintPoints();
        buildConnectivity();
        solveTensorField();
        buildGridTensors();
    }

    template <typename Scalar>
    void TensorField2D<Scalar>::setTensorWeight(Scalar weight)
    {
        terrainWeight = weight;
        solveTensorField();
        buildGridTensors();
    }

    template <typename Scalar>
    void TensorField2D<Scalar>::resolveTensor()
    {
        solveTensorField();
        buildGridTensors();
    }

    template <typename Scalar>
    void TensorField2D<Scalar>::buildGrid()
    {
        using Vector2 = Vector2<Scalar>;
        Scalar w = bound_.sizes().x();
        Scalar h = bound_.sizes().y();
        // std::cout << "AABB width: " << w << ", height: " << h << std::endl;

        gridSize_ = std::min(w, h) / Scalar(minGridNum_);
        nx_ = int(std::ceil(w / gridSize_));
        ny_ = int(std::ceil(h / gridSize_));

        gridPoints_.clear();
        gridPoints_.reserve((nx_ + 1) * (ny_ + 1));
        for (int j = 0; j <= ny_; ++j)
        {
            for (int i = 0; i <= nx_; ++i)
            {
                Vector2 p;
                p.x() = bound_.min().x() + i * gridSize_;
                p.y() = bound_.min().y() + j * gridSize_;
                gridPoints_.push_back(p);
            }
        }
        // std::cout << "Grid size: " << gridSize_ << ", nx: " << nx_ << ", ny: " << ny_ << std::endl;
        // std::cout << "Total grid points: " << gridPoints_.size() << std::endl;
    }

    template <typename Scalar>
    void TensorField2D<Scalar>::collectConstraintPoints()
    {
        using Vector2 = Vector2<Scalar>;
        allPoints_ = gridPoints_;
        ptLabels.resize(allPoints_.size(), 0);
        auto add_constraint = [&](const Vector2 &p, const Vector2 &dir)
        {
            int idx = (int)allPoints_.size();
            allPoints_.push_back(p);
            ptLabels.push_back(1); // constraint point
            fixedThethas_[idx] = directionToTheta<Scalar>(dir);
        };

        for (const auto &poly : polylines_)
        {
            for (int i = 0; i + 1 < (int)poly.points.size(); ++i)
            {
                Vector2 p1 = poly.points[i];
                Vector2 p2 = poly.points[i + 1];
                Vector2 t = (p2 - p1).normalized();

                // 与竖线
                for (int ix = 0; ix <= nx_; ++ix)
                {
                    Scalar x = bound_.min().x() + ix * gridSize_;
                    if ((p1.x() - x) * (p2.x() - x) < 0)
                    {
                        Scalar y = p1.y() + (x - p1.x()) * (p2.y() - p1.y()) / (p2.x() - p1.x());
                        add_constraint(Vector2(x, y), t);
                    }
                }

                // 与横线
                for (int iy = 0; iy <= ny_; ++iy)
                {
                    Scalar y = bound_.min().y() + iy * gridSize_;
                    if ((p1.y() - y) * (p2.y() - y) < 0)
                    {
                        Scalar x = p1.x() + (y - p1.y()) * (p2.x() - p1.x()) / (p2.y() - p1.y());
                        add_constraint(Vector2(x, y), t);
                    }
                }
            }
        }
        // std::cout << "Total grid points: " << allPoints_.size() << std::endl;
    }

    template <typename Scalar>
    inline Scalar attractorWeight(Scalar dist2, Scalar radius)
    {
        // Gaussian / RBF
        // w = exp(-d^2 / r^2)
        return std::exp(-dist2 / (radius * radius));
    }

    template <typename Scalar>
    void TensorField2D<Scalar>::buildConnectivity()
    {
        edges_.clear();

        PointCloud2D<Scalar> cloud(allPoints_);

        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<Scalar, PointCloud2D<Scalar>>, PointCloud2D<Scalar>, 2>;

        KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();

        const Scalar radius = gridSize_ * Scalar(1.2);
        const Scalar radius2 = radius * radius;

        nanoflann::SearchParameters params;
        std::vector<nanoflann::ResultItem<uint32_t, Scalar>> matches;

        for (size_t i = 0; i < allPoints_.size(); ++i)
        {
            matches.clear();
            const Scalar query_pt[2] = {
                allPoints_[i].x(),
                allPoints_[i].y()};

            tree.radiusSearch(query_pt, radius2, matches, params);

            for (auto &m : matches)
            {
                size_t j = m.first;
                if (j == i)
                    continue;

                // 防止重复边（i < j）
                if (i < j)
                    edges_.emplace_back((int)i, (int)j);
            }
        }
    }

    template <typename Scalar>
    void TensorField2D<Scalar>::solveTensorField()
    {
        using SparseMat = Eigen::SparseMatrix<Scalar>;
        using Triplet = Eigen::Triplet<Scalar>;
        using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const int N = static_cast<int>(allPoints_.size());
        const int num_vars = 2 * N; // sin + cos

        // ===============================
        // 1. 构建 A 和 b
        // ===============================
        std::vector<Triplet> trips;
        Vec b(num_vars);
        b.setZero();

        // -------- Laplacian--------

        // A[i,i] = n
        // A[i,j] = -1
        for (auto &e : edges_)
        {
            int i = e.first;
            int j = e.second;

            // sin block
            trips.emplace_back(i, i, 1);
            trips.emplace_back(i, j, -1);
            trips.emplace_back(j, j, 1);
            trips.emplace_back(j, i, -1);

            // cos block（整体偏移 N）
            trips.emplace_back(i + N, i + N, 1);
            trips.emplace_back(i + N, j + N, -1);
            trips.emplace_back(j + N, j + N, 1);
            trips.emplace_back(j + N, i + N, -1);
        }

        // ---------- (2) 吸引点软约束 ----------
        for (const auto &attr : attractors_)
        {
            for (int i = 0; i < N; ++i)
            {
                // 如果是硬约束点，直接跳过
                if (ptLabels[i] == 1)
                    continue;

                Vector2<Scalar> p = allPoints_[i];
                Vector2<Scalar> dir = attr.pos - p;
                Scalar dist2 = dir.squaredNorm();

                if (dist2 > attr.radius * attr.radius)
                    continue;

                dir.normalize();
                Scalar theta_attr = directionToTheta<Scalar>(dir);

                Scalar sin4 = std::sin(Scalar(4) * theta_attr);
                Scalar cos4 = std::cos(Scalar(4) * theta_attr);

                Scalar w = attractorWeight(dist2, attr.radius);

                // sin
                trips.emplace_back(i, i, w);
                b[i] += w * sin4;

                // cos
                trips.emplace_back(i + N, i + N, w);
                b[i + N] += w * cos4;
            }
        }
        // ---------- (3) 地形 tensor 软约束 ----------
        for (int i = 0; i < gridPoints_.size(); ++i)
        {
            TerrainTensor terrainT = terrainTensors_[i];
            if (terrainT.init < 0)
                continue;

            // 坡向 → theta（定义的是下坡方向）
            Scalar theta_t = terrainT.aspect;

            Scalar sin4 = std::sin(Scalar(4) * theta_t);
            Scalar cos4 = std::cos(Scalar(4) * theta_t);

            // ===== 权重设计（关键）=====
            // slope 越大，方向性越强
            // slope ∈ [0, pi/2]，归一化
            Scalar slope01 = terrainT.slope / Scalar(0.5 * M_PI);
            slope01 = std::clamp(slope01, Scalar(0), Scalar(1));
            // if (slope01 < Scalar(1e-6))
            //     continue;
            Scalar w = terrainWeight * slope01;
            // terrainWeight_ 建议是 1 ~ 10，可调参数

            if (w < Scalar(1e-6))
                continue;

            // sin block
            trips.emplace_back(i, i, w);
            b[i] += w * sin4;

            // cos block
            trips.emplace_back(i + N, i + N, w);
            b[i + N] += w * cos4;
        }

        SparseMat A(num_vars, num_vars);
        A.setFromTriplets(trips.begin(), trips.end());
        A.makeCompressed();

        // ===============================
        // 2. 固定点（硬约束）
        // ===============================
        // if label[i]:
        //   A[row] = 0
        //   A[ii] = 1
        //   b = sin/cos(4θ)

        for (int i = 0; i < N; ++i)
        {
            if (ptLabels[i] == 1) // 固定点
            {
                Scalar t = fixedThethas_[i];
                Scalar s = std::sin(Scalar(4) * t);
                Scalar c = std::cos(Scalar(4) * t);

                int sin_idx = i;
                int cos_idx = i + N;

                // ---- 清空 sin 行 ----
                for (typename SparseMat::InnerIterator it(A, sin_idx); it; ++it)
                    it.valueRef() = 0;
                A.coeffRef(sin_idx, sin_idx) = 1;
                b[sin_idx] = s;

                // ---- 清空 cos 行 ----
                for (typename SparseMat::InnerIterator it(A, cos_idx); it; ++it)
                    it.valueRef() = 0;
                A.coeffRef(cos_idx, cos_idx) = 1;
                b[cos_idx] = c;
            }
        }

        A.makeCompressed();

        // ===============================
        // 3. 初始值 x0
        // ===============================
        Vec x0(num_vars);
        for (int i = 0; i < N; ++i)
        {
            Scalar t = fixedThethas_[i];
            x0[i] = std::sin(Scalar(4) * t);
            x0[i + N] = std::cos(Scalar(4) * t);
        }

        // ===============================
        // 4. 求解（CG）
        // ===============================
        Eigen::ConjugateGradient<SparseMat, Eigen::Lower | Eigen::Upper> cg;
        cg.setMaxIterations(100);
        cg.compute(A);

        Vec x = cg.solveWithGuess(b, x0);

        // ===============================
        // 5. 恢复 theta
        // ===============================
        theta_.resize(N);
        allTensors_.clear();
        // gridPtTensors_.clear();
        allTensors_.reserve(N);
        // gridPtTensors_.reserve(gridPoints_.size());

        for (int i = 0; i < N; ++i)
        {
            Scalar sin4 = x[i];
            Scalar cos4 = x[i + N];

            Scalar theta = Scalar(0.25) * std::atan2(sin4, cos4);
            if (theta < 0)
                theta += Scalar(0.5 * M_PI);

            theta_[i] = theta;

            Tensor<Scalar> tensor(allPoints_[i], theta);
            allTensors_.push_back(tensor);

            // if (ptLabels[i] == 1)
            //     gridPtTensors_.push_back(tensor);
        }
    }

    template <typename Scalar>
    void TensorField2D<Scalar>::buildGridTensors()
    {
        gridTensors_.clear();
        gridTensors_.reserve((nx_ * ny_));
        cellCenters_.clear();
        cellCenters_.reserve((nx_ * ny_));

        auto idx = [&](int ix, int iy)
        {
            return iy * (nx_ + 1) + ix;
        };

        Vector2<Scalar> minP = bound_.min();

        for (int iy = 0; iy < ny_; ++iy)
        {
            for (int ix = 0; ix < nx_; ++ix)
            {
                int i01 = idx(ix, iy);         // 左下01
                int i11 = idx(ix + 1, iy);     // 右下11
                int i00 = idx(ix, iy + 1);     // 左上00
                int i10 = idx(ix + 1, iy + 1); // 右上10

                GridTensor<Scalar> cell;
                cell.center = Vector2<Scalar>(minP.x() + (ix + Scalar(0.5)) * gridSize_, minP.y() + (iy + Scalar(0.5)) * gridSize_);

                cell.theta[0] = theta_[i00];
                cell.theta[1] = theta_[i10];
                cell.theta[2] = theta_[i01];
                cell.theta[3] = theta_[i11];

                gridTensors_.push_back(cell);
                cellCenters_.push_back(cell.center);
            }
        }
        // std::cout << "Built grid tensors: " << gridTensors_.size() << std::endl;
    }

    template <typename Scalar>
    std::array<Vector2<Scalar>, 4> TensorField2D<Scalar>::getTensorAt(const Vector2<Scalar> &pos) const
    {
        Vector2<Scalar> minP = bound_.min();

        Scalar fx = (pos.x() - minP.x()) / gridSize_;
        Scalar fy = (pos.y() - minP.y()) / gridSize_;

        int ix = std::clamp(static_cast<int>(std::floor(fx)), 0, nx_ - 1);
        int iy = std::clamp(static_cast<int>(std::floor(fy)), 0, ny_ - 1);

        int cellId = iy * nx_ + ix;
        // std::cout<<"cellId: " << cellId << std::endl;
        const GridTensor<Scalar> &cell = gridTensors_[cellId];

        Scalar u = fx - ix;
        Scalar v = 1 - fy + iy;

        Scalar a00 = cell.theta[0];
        Scalar a10 = cell.theta[1];
        Scalar a01 = cell.theta[2];
        Scalar a11 = cell.theta[3];

        Scalar sin4 =
            std::sin(Scalar(4) * a00) * (1 - u) * (1 - v) +
            std::sin(Scalar(4) * a10) * (u) * (1 - v) +
            std::sin(Scalar(4) * a01) * (1 - u) * (v) +
            std::sin(Scalar(4) * a11) * (u) * (v);

        Scalar cos4 =
            std::cos(Scalar(4) * a00) * (1 - u) * (1 - v) +
            std::cos(Scalar(4) * a10) * (u) * (1 - v) +
            std::cos(Scalar(4) * a01) * (1 - u) * (v) +
            std::cos(Scalar(4) * a11) * (u) * (v);

        Scalar theta = Scalar(0.25) * std::atan2(sin4, cos4);

        Vector2<Scalar> v0(std::cos(theta), std::sin(theta));
        Vector2<Scalar> v1(std::sin(theta), -std::cos(theta));
        Vector2<Scalar> v2(-v0.x(), -v0.y());
        Vector2<Scalar> v3(-v1.x(), -v1.y());

        return {v0, v1, v2, v3};
    }

    template <typename Scalar>
    Tensor<Scalar> TensorField2D<Scalar>::testTensorAt(const Vector2<Scalar> &pos) const
    {
        Vector2<Scalar> minP = bound_.min();

        Scalar fx = (pos.x() - minP.x()) / gridSize_;
        Scalar fy = (pos.y() - minP.y()) / gridSize_;

        int ix = std::clamp(static_cast<int>(std::floor(fx)), 0, nx_ - 1);
        int iy = std::clamp(static_cast<int>(std::floor(fy)), 0, ny_ - 1);

        int cellId = iy * nx_ + ix;
        std::cout << "cellId: " << cellId << std::endl;
        const GridTensor<Scalar> &cell = gridTensors_[cellId];

        Scalar u = fx - ix;
        Scalar v = 1 - fy + iy;

        std::cout << "u: " << u << ", v: " << v << std::endl;

        Scalar a00 = cell.theta[0];
        Scalar a10 = cell.theta[1];
        Scalar a01 = cell.theta[2];
        Scalar a11 = cell.theta[3];

        Scalar sin4 =
            std::sin(Scalar(4) * a00) * (1 - u) * (1 - v) +
            std::sin(Scalar(4) * a10) * (u) * (1 - v) +
            std::sin(Scalar(4) * a01) * (1 - u) * (v) +
            std::sin(Scalar(4) * a11) * (u) * (v);

        Scalar cos4 =
            std::cos(Scalar(4) * a00) * (1 - u) * (1 - v) +
            std::cos(Scalar(4) * a10) * (u) * (1 - v) +
            std::cos(Scalar(4) * a01) * (1 - u) * (v) +
            std::cos(Scalar(4) * a11) * (u) * (v);

        Scalar theta = Scalar(0.25) * std::atan2(sin4, cos4);

        return Tensor<Scalar>(pos, theta);
    }

    template <typename Scalar>
    Polyline2_t<Scalar> TensorField2D<Scalar>::traceStreamlineFromPos(const Vector2<Scalar> &pos, int uv, int maxIter) const
    {
        Polyline2_t<Scalar> poly;
        poly.points.reserve(maxIter * 2 + 1);
        Scalar step = gridSize_ / Scalar(8);

        int i = (uv == 0) ? 0 : 1;
        int j = (uv == 0) ? 2 : 3;

        // 初始点
        Vector2<Scalar> current = pos;
        poly.points.push_back(current);

        // 初始 tensor
        auto tensors = getTensorAt(current);
        Vector2<Scalar> dir1 = tensors[i];
        Vector2<Scalar> dir2 = tensors[j];

        Vector2<Scalar> forward = current + dir1 * step;
        Vector2<Scalar> backward = current + dir2 * step;

        // === 初始插点===
        if (isInside(forward))
            poly.points.push_back(forward);

        if (isInside(backward))
            poly.points.insert(poly.points.begin(), backward);

        // ======================================================
        // 前向追踪
        // ======================================================
        {
            current = forward;
            Vector2<Scalar> preDir = dir1;

            int iter = 0;
            while (isInside(current) && iter < maxIter)
            {
                auto ts = getTensorAt(current);

                Vector2<Scalar> best = ts[0];
                Scalar maxDot = best.dot(preDir);

                for (const auto &v : ts)
                {
                    Scalar d = v.dot(preDir);
                    if (d > maxDot)
                    {
                        maxDot = d;
                        best = v;
                    }
                }

                Vector2<Scalar> next = current + best * step;
                if (!isInside(next))
                    break;

                poly.points.push_back(next);
                current = next;
                preDir = best;
                ++iter;
            }
        }

        // ======================================================
        // 反向追踪
        // ======================================================
        {
            current = backward;
            Vector2<Scalar> preDir = dir2;

            int iter = 0;
            while (isInside(current) && iter < maxIter)
            {
                auto ts = getTensorAt(current);

                Vector2<Scalar> best = ts[0];
                Scalar maxDot = best.dot(preDir);

                for (const auto &v : ts)
                {
                    Scalar d = v.dot(preDir);
                    if (d > maxDot)
                    {
                        maxDot = d;
                        best = v;
                    }
                }

                Vector2<Scalar> next = current + best * step;
                if (!isInside(next))
                    break;

                poly.points.insert(poly.points.begin(), next);
                current = next;
                preDir = best;
                ++iter;
            }
        }

        return poly;
    }

    template <typename Scalar>
    Polyline2_t<Scalar> TensorField2D<Scalar>::traceStreamlineFromPosWithDir(const Vector2<Scalar> &pos, const Vector2<Scalar> &startDir, Scalar step, int maxIter) const
    {
        Polyline2_t<Scalar> poly;
        // 初始点
        Vector2<Scalar> current = pos;
        poly.points.push_back(current);

        Vector2<Scalar> preDir = startDir;

        int iter = 0;
        while (isInside(current) && iter < maxIter)
        {
            auto ts = getTensorAt(current);

            Vector2<Scalar> best = ts[0];
            Scalar maxDot = best.dot(preDir);

            for (const auto &v : ts)
            {
                Scalar d = v.dot(preDir);
                if (d > maxDot)
                {
                    maxDot = d;
                    best = v;
                }
            }

            Vector2<Scalar> next = current + best * step;
            if (!isInside(next))
                break;

            poly.points.push_back(next);
            current = next;
            preDir = best;
            ++iter;
        }

        return poly;
    }

    template <typename Scalar>
    std::vector<Polyline2_t<Scalar>> TensorField2D<Scalar>::genStreamlines(const std::vector<Vector2<Scalar>> &points, int maxIter) const
    {

        std::vector<Polyline2_t<Scalar>> streamlines;
        for (const auto &p : points)
        {
            streamlines.push_back(traceStreamlineFromPos(p, 0, maxIter));
            streamlines.push_back(traceStreamlineFromPos(p, 1, maxIter));
        }
        return streamlines;
    }

    template <typename Scalar>
    inline Vector2<Scalar> thetaToDir(Scalar theta)
    {
        return Vector2<Scalar>(std::cos(theta), std::sin(theta));
    }

    template <typename Scalar>
    inline void thetaToCross(
        Scalar theta,
        Vector2<Scalar> &d0,
        Vector2<Scalar> &d1)
    {
        d0 = thetaToDir(theta);
        d1 = Vector2<Scalar>(-d0.y(), d0.x());
    }

    template <typename Scalar>
    Polyline2_t<Scalar> createRandomPolygon(int ptNum, Scalar scale, Scalar threshold, const Vector2<Scalar> &base)
    {
        Polyline2_t<Scalar> poly;
        poly.isClosed = true;

        if (ptNum < 3)
            return poly;

        // === 与 Java 完全一致的 threshold 修正 ===
        Scalar thre = (threshold >= Scalar(1) || threshold <= Scalar(0)) ? Scalar(0.5) : threshold;

        Scalar angleStep = Scalar(2) * Scalar(M_PI) / Scalar(ptNum);

        poly.points.reserve(ptNum + 1);

        for (int i = 0; i < ptNum; ++i)
        {
            // Math.random() ∈ [0,1)
            Scalar r01 = Scalar(std::rand()) / Scalar(RAND_MAX);

            // (Math.random() * 2 * thre + 1 - thre) * scale
            Scalar ran = (r01 * Scalar(2) * thre + Scalar(1) - thre) * scale;

            Scalar a = angleStep * Scalar(i);

            Vector2<Scalar> p;
            p.x() = base.x() + std::cos(a) * ran;
            p.y() = base.y() + std::sin(a) * ran;

            poly.points.push_back(p);
        }

        // === 显式闭合（WB_Polygon 的等价行为）===
        poly.points.push_back(poly.points.front());

        return poly;
    }

    template <typename Scalar>
    inline AABB<Scalar> computeAABB(const std::vector<Polyline2_t<Scalar>> &polylines)
    {
        AABB<Scalar> box;
        box.setEmpty(); // 非常重要

        for (const auto &poly : polylines)
        {
            for (const auto &p : poly.points)
            {
                box.extend(p);
            }
        }
        // std::cout << "AABB min: (" << box.min().x() << "," << box.min().y() << "), max: (" << box.max().x() << "," << box.max().y() << ")" << std::endl;
        return box;
    }

    template <typename Scalar>
    Polyline2_t<Scalar> rebuildPolyline(const Polyline2_t<Scalar> &polyline, Scalar threshold)
    {
        Polyline2_t<Scalar> result;
        result.isClosed = polyline.isClosed;

        const auto &pts = polyline.points;
        if (pts.size() < 2 || threshold <= Scalar(0))
        {
            result.points = pts;
            return result;
        }

        const int segCount = static_cast<int>(pts.size()) - 1;

        for (int i = 0; i < segCount; ++i)
        {
            const Vector2<Scalar> &p0 = pts[i];
            const Vector2<Scalar> &p1 = pts[(i + 1) % pts.size()];

            // 先压入起点
            result.points.push_back(p0);

            Scalar dist = (p1 - p0).norm();
            int n = static_cast<int>(std::floor(dist / threshold));

            if (n > 1)
            {
                // 均匀插值
                for (int k = 1; k < n; ++k)
                {
                    Scalar t = Scalar(k) / Scalar(n);
                    Vector2<Scalar> p = (Scalar(1) - t) * p0 + t * p1;
                    result.points.push_back(p);
                }
            }
        }

        // 非闭合 polyline，补最后一个点
        if (!polyline.isClosed)
        {
            result.points.push_back(pts.back());
        }

        return result;
    }

    template <typename Scalar>
    std::unordered_map<int, std::vector<Polyline2_t<Scalar>>>
    TensorField2D<Scalar>::traceCrossLinesBetweenGaps(
        int seed,
        int dir,
        Scalar rebuildStep,
        Scalar traceStep,
        const Eigen::Vector2i &gapRange) const
    {
        std::unordered_map<int, std::vector<Polyline2_t<Scalar>>> result;
        int globalIdx = 0;

        for (int pid = 0; pid < (int)polylines_.size(); ++pid)
        {
            Polyline2_t<Scalar> poly = rebuildPolyline(polylines_[pid], rebuildStep);
            const int nPts = (int)poly.points.size();

            if (nPts < gapRange.x())
                continue;

            std::mt19937 rng(seed + pid);
            std::uniform_int_distribution<int> gapDist(gapRange.x(), gapRange.y());

            int idx = 0;
            int segId = globalIdx;

            while (idx < nPts)
            {
                int gap = gapDist(rng);
                int remain = nPts - idx;

                // 尾段太短，直接结束
                if (remain < gapRange.x() / 2)
                    break;

                const int end = std::min(idx + gap, nPts);
                // ===== 根据 gap 自适应生成 depth =====
                int depthMin = std::max(static_cast<int>(std::ceil((Scalar)gap * Scalar(2.0 / 3.0))), gapRange.x());
                int depthMax = static_cast<int>(std::floor((Scalar)gap * Scalar(4.0 / 3.0)));

                // 防御：避免 min > max
                if (depthMin > depthMax)
                    std::swap(depthMin, depthMax);

                std::uniform_int_distribution<int> depthDist(depthMin, depthMax);
                int maxIter = depthDist(rng);
                // === 核心：同一个 gap 内的每一个点都 trace ===
                for (int i = idx; i < end; ++i)
                {
                    const Vector2<Scalar> &startPos = poly.points[i];

                    Vector2<Scalar> tangent = poly.getTangentAt(i);
                    Vector2<Scalar> startDir = (dir == 0) ? geo::rotate90CCW(tangent) : geo::rotate90CW(tangent);
                    Polyline2_t<Scalar> traced = traceStreamlineFromPosWithDir(startPos, startDir, traceStep, maxIter);

                    if (traced.points.size() > 1)
                    {
                        result[segId].push_back(traced);
                    }
                }

                idx += gap;
                ++segId;
            }

            globalIdx = segId;
        }

        return result;
    }

    template <typename Scalar>
    std::vector<Polyline2_t<Scalar>> buildParcelFromTraceStrips(const std::unordered_map<int, std::vector<Polyline2_t<Scalar>>> &stripMap)
    {
        std::vector<Polyline2_t<Scalar>> parcels;

        for (const auto &[segId, traces] : stripMap)
        {
            const int n = (int)traces.size();
            if (n < 2)
                continue;

            Polyline2_t<Scalar> parcel;
            parcel.isClosed = true;
            const auto &first = traces.front(); // first polyline
            const auto &last = traces.back();   // last polyline
            if (first.points.size() < 2 || last.points.size() < 2)
                continue;

            // === 1. 起点序列（上边界） ===
            for (const auto &pl : traces)
            {
                if (!pl.points.empty())
                    parcel.points.push_back(pl.points.front());
            }

            // === 2. 最后一根 polyline（右边界，正序，去首） ===
            for (int i = 1; i < (int)last.points.size(); ++i)
                parcel.points.push_back(last.points[i]);

            // === 3. 终点序列（下边界，reverse + 方向过滤） ===
            for (int i = n - 2; i >= 0; --i)
            {
                if (traces[i].points.empty())
                    continue;

                const Vector2<Scalar> &p = traces[i].points.back();

                if (parcel.points.size() >= 2)
                {
                    Vector2<Scalar> prevDir = parcel.points.back() - parcel.points[parcel.points.size() - 2];
                    Vector2<Scalar> newDir = p - parcel.points.back();
                    // prevDir.normalize();
                    // newDir.normalize();
                    if (prevDir.dot(newDir) < Scalar(0))
                        continue; // 方向反了，跳过
                }

                parcel.points.push_back(p);
            }

            Vector2<Scalar> prevDir = first.points[first.points.size() - 2] - parcel.points.back();
            Vector2<Scalar> newDir = first.points[first.points.size() - 3] - first.points[first.points.size() - 2];
            if (prevDir.dot(newDir) < Scalar(0))
                continue;
            // === 4. 第一根 polyline（左边界，reverse，去首） ===
            for (int i = (int)first.points.size() - 2; i >= 0; --i)
                parcel.points.push_back(first.points[i]);

            Scalar parcelArea = util::Math2<Scalar>::polygon_area(parcel.points);
            if (parcelArea < Scalar(150))
                continue;
            //     std::reverse(parcel.points.begin(), parcel.points.end());

            geo::OBB2<Scalar> obb(parcel.points);
            Scalar minWidth = std::min(obb.halfSize.x(), obb.halfSize.y());
            Scalar maxWidth = std::max(obb.halfSize.x(), obb.halfSize.y());

            Scalar obbArea = util::Math2<Scalar>::polygon_area(obb.poly.points);

            if ((maxWidth / minWidth > Scalar(2.3) && minWidth <= 20) || obbArea / parcelArea >= Scalar(1.85))
                continue;

            if (!parcel.isSelfIntersecting())
                parcels.push_back(std::move(parcel));
        }

        return parcels;
    }

    template <typename Scalar>
    std::vector<Polyline2_t<Scalar>> TensorField2D<Scalar>::trace2sidesParcels(int seed, Scalar rebuildStep, Scalar traceStep, const Eigen::Vector2i &gapRange/* , Scalar cellSize, int width, int height */) const
    {
        using AABB = Eigen::AlignedBox<Scalar, 2>;
        std::unordered_map<int, std::vector<Polyline2_t<Scalar>>> result_1 = traceCrossLinesBetweenGaps(seed, 0, rebuildStep, traceStep, gapRange);
        std::unordered_map<int, std::vector<Polyline2_t<Scalar>>> result_2 = traceCrossLinesBetweenGaps(seed + 1314, 1, rebuildStep, traceStep, gapRange);
        std::vector<Polyline2_t<Scalar>> parcels = buildParcelFromTraceStrips(result_1);
        std::vector<Polyline2_t<Scalar>> parcels2 = buildParcelFromTraceStrips(result_2);
        parcels.insert(
            parcels.end(),
            std::make_move_iterator(parcels2.begin()),
            std::make_move_iterator(parcels2.end()));

        // 去重叠的parcels
        // std::vector<std::vector<uint32_t>> gridInsideParcels;
        // gridInsideParcels.reserve((width + 1) * (height + 1));
        // for (int j = 0; j <= height; j++)
        // {
        //     for (int i = 0; i <= width; i++)
        //     {
        //         Vector2<Scalar> pos(i * cellSize - width / 2 * cellSize, j * cellSize - height / 2 * cellSize);
        //         for (int k = 0; k < parcels.size(); k++)
        //         {
        //             const Polyline2_t<Scalar> parcel = parcels[k];
        //             AABB bound = parcel.getAABB2();
        //             if (bound.contains(pos))
        //                 gridInsideParcels[j * (width + 1) + i] = k;
        //         }
        //     }
        // }
        return parcels;
    }

    void buildPlotUISettings();
}