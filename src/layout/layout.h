#pragma once

#include "terrain.h"
#include "geo.h"
#include "diffVoronoi.h"
#include <torch/torch.h>

namespace layout
{

    template <typename Scalar>
    struct RoomPoint2D
    {
        Eigen::Vector2<Scalar> pos; // 局部坐标
        std::vector<int> storeys;   // 楼层（可选）
        RoomPoint2D() = default;
        RoomPoint2D(const Eigen::Vector2<Scalar> &p) : pos(p) {}
    };

    struct SeparateRegion
    {
        std::vector<int> left;
        std::vector<int> right;
        std::vector<int> front;
        std::vector<int> back;
        SeparateRegion() = default;
    };

    struct RVDNeighbors
    {
        std::vector<std::vector<int>> left, right, front, back;
        RVDNeighbors() = default;
    };

    template <typename Scalar>
    struct RectCell
    {
        Eigen::Vector2<Scalar> center; // 局部坐标
        Eigen::Vector2<Scalar> half;   // half width / height（局部）
        RectCell() = default;
        RectCell(const Eigen::Vector2<Scalar> &c) : center(c), half(0, 0) {}
        RectCell(const Eigen::Vector2<Scalar> &c, const Eigen::Vector2<Scalar> &h) : center(c), half(h) {}
    };

    template <typename Scalar>
    class RectVoronoi2D
    {
    public:
        RectVoronoi2D(const std::vector<RoomPoint2D<Scalar>> &points, const Eigen::AlignedBox<Scalar, 2> &bound);
        const std::vector<RectCell<Scalar>> &getCells() const { return cells; }
        const std::vector<geo::Polyline2_t<Scalar>> &getCellPolys() const { return cellPolys; }
        const RVDNeighbors &getNeighbors() const { return nbr; }

    private:
        void build();
        SeparateRegion computeSeparateRegion(int idx) const;
        RectCell<Scalar> computeCell(int idx, const SeparateRegion &region) const;
        geo::Polyline2_t<Scalar> buildCellPoly(const RectCell<Scalar> &cell) const;

    private:
        std::vector<RoomPoint2D<Scalar>> pts;
        Eigen::AlignedBox<Scalar, 2> bounds; // rect-local bounding box
        std::vector<RectCell<Scalar>> cells;
        std::vector<geo::Polyline2_t<Scalar>> cellPolys;
        RVDNeighbors nbr;
    };

    template <typename Scalar>
    RectVoronoi2D<Scalar>::RectVoronoi2D(const std::vector<RoomPoint2D<Scalar>> &points, const Eigen::AlignedBox<Scalar, 2> &bound)
        : pts(points), bounds(bound)
    {
        build();
    }

    template <typename Scalar>
    SeparateRegion RectVoronoi2D<Scalar>::computeSeparateRegion(int idx) const
    {
        SeparateRegion r;
        const auto &p = pts[idx].pos;

        for (int i = 0; i < pts.size(); ++i)
        {
            if (i == idx)
                continue;
            const auto &q = pts[i].pos;

            if (p.x() - q.x() > std::abs(p.y() - q.y()))
                r.left.push_back(i);
            else if (q.x() - p.x() > std::abs(p.y() - q.y()))
                r.right.push_back(i);
            else if (p.y() - q.y() > std::abs(p.x() - q.x()))
                r.front.push_back(i);
            else if (q.y() - p.y() > std::abs(p.x() - q.x()))
                r.back.push_back(i);
        }
        return r;
    }

    template <typename Scalar>
    RectCell<Scalar>
    RectVoronoi2D<Scalar>::computeCell(int idx, const SeparateRegion &region) const
    {
        const auto &p = pts[idx].pos;

        Scalar l = bounds.min().x();
        Scalar r = bounds.max().x();
        Scalar f = bounds.min().y();
        Scalar b = bounds.max().y();

        // Left: x >= (p.x + q.x) / 2
        for (int qi : region.left)
        {
            Scalar mid = (p.x() + pts[qi].pos.x()) * Scalar(0.5);
            l = std::max(l, mid);
        }

        // Right: x <= (p.x + q.x) / 2
        for (int qi : region.right)
        {
            Scalar mid = (p.x() + pts[qi].pos.x()) * Scalar(0.5);
            r = std::min(r, mid);
        }

        // Front: y >= (p.y + q.y) / 2
        for (int qi : region.front)
        {
            Scalar mid = (p.y() + pts[qi].pos.y()) * Scalar(0.5);
            f = std::max(f, mid);
        }

        // Back: y <= (p.y + q.y) / 2
        for (int qi : region.back)
        {
            Scalar mid = (p.y() + pts[qi].pos.y()) * Scalar(0.5);
            b = std::min(b, mid);
        }

        RectCell<Scalar> cell;
        cell.center = {(l + r) * Scalar(0.5),
                       (f + b) * Scalar(0.5)};
        cell.half = {(r - l) * Scalar(0.5),
                     (b - f) * Scalar(0.5)};
        return cell;
    }

    template <typename Scalar>
    void RectVoronoi2D<Scalar>::build()
    {
        cells.clear();
        cellPolys.clear();
        const int N = static_cast<int>(pts.size());
        cells.reserve(N);
        cellPolys.reserve(N);

        nbr.left.resize(N);
        nbr.right.resize(N);
        nbr.front.resize(N);
        nbr.back.resize(N);
        for (int i = 0; i < N; ++i)
        {
            SeparateRegion region = computeSeparateRegion(i);
            nbr.left[i] = region.left;
            nbr.right[i] = region.right;
            nbr.front[i] = region.front;
            nbr.back[i] = region.back;
            RectCell<Scalar> cell = computeCell(i, region);

            cells.push_back(cell);
            cellPolys.push_back(buildCellPoly(cell));
        }
    }

    template <typename Scalar>
    geo::Polyline2_t<Scalar> RectVoronoi2D<Scalar>::buildCellPoly(const RectCell<Scalar> &cell) const
    {
        const auto &c = cell.center;
        const auto &h = cell.half;

        std::vector<Eigen::Vector2<Scalar>> pts;
        pts.reserve(5);

        pts.emplace_back(c.x() - h.x(), c.y() - h.y());
        pts.emplace_back(c.x() + h.x(), c.y() - h.y());
        pts.emplace_back(c.x() + h.x(), c.y() + h.y());
        pts.emplace_back(c.x() - h.x(), c.y() + h.y());
        pts.push_back(pts.front()); // close

        return geo::Polyline2_t<Scalar>(pts, true);
    }

    template <typename Scalar>
    class BuildingLayout
    {
        template <typename Scalar>
        using Polyline2_t = geo::Polyline2_t<Scalar>;
        using Point_3 = geo::Point_3;
        using SurfaceMesh = geo::SurfaceMesh;
        template <typename Scalar>
        using Vector2 = Eigen::Matrix<Scalar, 2, 1>;

    public:
        BuildingLayout() = default;
        BuildingLayout(const Polyline2_t<Scalar> &site_, const terrain::Terrain &terrain);
        void upload() { model = LoadModelFromMesh(geo::buildRaylibMesh(meshData)); }
        void drawTerrain(Color color, float colorAlpha, bool wireframe, float wireframeAlpha) const;

    private:
        void initLayout(const terrain::Terrain &terrain);

    private:
        geo::MeshData meshData; // CPU mesh
        Model model;
        // GPU mesh
    public:
        Polyline2_t<Scalar> site;
        Polyline2_t<Scalar> oriRect;
        Polyline2_t<Scalar> rotedRect;
        Eigen::AlignedBox<Scalar, 2> rotedBound;
        Eigen::Matrix<Scalar, 2, 2> Rinv;
        std::vector<Scalar> heightMap;
        std::vector<Vector2<Scalar>> rotedCenters;
        Scalar divGap;
        // std::vector<Vector2<Scalar>> rotedVertices;
        // std::vector<Eigen::Vector3<Scalar>> rotedGrids;
        SurfaceMesh heMesh;
    };

    template <typename Scalar>
    BuildingLayout<Scalar>::BuildingLayout(const geo::Polyline2_t<Scalar> &site_, const terrain::Terrain &terrain)
        : site(site_)
    {
        initLayout(terrain);
        upload();
    }

    template <typename Scalar>
    void BuildingLayout<Scalar>::initLayout(const terrain::Terrain &terrain)
    {
        // 1. compute OBB of site
        geo::OBB2<Scalar> obb(site.points);
        Eigen::Matrix<Scalar, 2, 2> R = geo::rotationToXAxis(obb.axis0);
        Rinv = R.transpose();
        Polyline2_t<Scalar> rotPoly = geo::rotatePoly(site, R);
        // --- compute max rect in rotated space
        rotedRect = geo::getMaxRectInPoly(rotPoly, 0.5);
        std::vector<Eigen::Vector2<Scalar>> originalPts;
        for (const auto &p : rotedRect.points)
            originalPts.emplace_back(Rinv * p);

        oriRect = Polyline2_t<Scalar>(originalPts, true);
        // 2. sample height for each vertex
        heightMap.clear();
        meshData.vertices.clear();
        meshData.indices.clear();
        Vector2<Scalar> p0 = oriRect.points[0];
        Vector2<Scalar> p1 = oriRect.points[1];
        Vector2<Scalar> p3 = oriRect.points[3];
        Vector2<Scalar> edgeW = p1 - p0; // 宽方向
        Vector2<Scalar> edgeH = p3 - p0; // 高方向
        Scalar width = edgeW.norm();
        Scalar height = edgeH.norm();
        Scalar newW = std::floor(width);
        Scalar newH = std::floor(height);
        Vector2<Scalar> center = (p0 + oriRect.points[2]) * Scalar(0.5);
        Vector2<Scalar> dirW = edgeW.normalized();
        Vector2<Scalar> dirH = edgeH.normalized();
        Scalar hw = newW * Scalar(0.5);
        Scalar hh = newH * Scalar(0.5);
        Scalar minW = std::min(newW, newH);
        Scalar maxW = std::max(newW, newH);
        divGap = std::min(Scalar(0.3 * minW), Scalar(0.2 * maxW));
        Vector2<Scalar> newP0 = center - dirW * hw - dirH * hh;
        Vector2<Scalar> newP1 = center + dirW * hw - dirH * hh;
        Vector2<Scalar> newP2 = center + dirW * hw + dirH * hh;
        Vector2<Scalar> newP3 = center - dirW * hw + dirH * hh;

        Vector2<Scalar> rotedP0 = R * newP0;
        Vector2<Scalar> rotedP1 = R * newP1;
        Vector2<Scalar> rotedP2 = R * newP2;
        Vector2<Scalar> rotedP3 = R * newP3;
        std::vector<Vector2<Scalar>> newRotedRect = {rotedP0, rotedP1, rotedP2, rotedP3, rotedP0};
        rotedRect = Polyline2_t<Scalar>(newRotedRect, true);

        rotedBound = geo::computeAABB(std::vector<Polyline2_t<Scalar>>{rotedRect});
        std::vector<Vector2<Scalar>> newRect = {newP0, newP1, newP2, newP3, newP0};
        oriRect = Polyline2_t<Scalar>(newRect, true);
        int nx = static_cast<int>(newW);
        int ny = static_cast<int>(newH);
        // build heMesh / meshData / renderMesh

        std::vector<std::vector<SurfaceMesh::Vertex_index>> vtx;

        vtx.resize(nx + 1);

        for (int i = 0; i <= nx; ++i)
        {
            vtx[i].resize(ny + 1);
            for (int j = 0; j <= ny; ++j)
            {
                Vector2<Scalar> v = newP0 + dirW * i + dirH * j;
                // rotedVertices.push_back(R * v);
                vtx[i][j] = heMesh.add_vertex(geo::to_cgal_point(v, Scalar(0)));
                Scalar height = Scalar(0);
                bool success = terrain.sampleHeightAt(height, v);
                // heightMap.push_back(height);
                geo::Vertex mv({(float)v.x(), (float)v.y(), (float)height});
                meshData.vertices.push_back(mv);
            }
        }
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                heMesh.add_face(
                    vtx[i][j],
                    vtx[i + 1][j],
                    vtx[i + 1][j + 1],
                    vtx[i][j + 1]);
                geo::Point_3 p00 = heMesh.point(vtx[i][j]);
                geo::Point_3 p11 = heMesh.point(vtx[i + 1][j + 1]);

                geo::Point_3 c3(
                    (p00.x() + p11.x()) * Scalar(0.5),
                    (p00.y() + p11.y()) * Scalar(0.5),
                    (p00.z() + p11.z()) * Scalar(0.5));

                Vector2<Scalar> center(c3.x(), c3.y());
                Vector2<Scalar> rotedCenter = R * center;
                rotedCenters.push_back(rotedCenter);

                Scalar height = Scalar(0);

                bool success = terrain.sampleHeightAt(height, center);
                heightMap.push_back(height);
                // std::cout<<"height is :"<< height <<std::endl;

                // rotedGrids.push_back({rotedCenter.x(),rotedCenter.y(), height});
                uint32_t i0 = i * (ny + 1) + j;
                uint32_t i1 = (i + 1) * (ny + 1) + j;
                uint32_t i2 = (i + 1) * (ny + 1) + (j + 1);
                uint32_t i3 = i * (ny + 1) + (j + 1);

                // triangle 1
                meshData.indices.push_back(i0);
                meshData.indices.push_back(i1);
                meshData.indices.push_back(i2);

                // triangle 2
                meshData.indices.push_back(i0);
                meshData.indices.push_back(i2);
                meshData.indices.push_back(i3);
            }
        }

        geo::computeVertexNormals(meshData);
    }

    template <typename Scalar>
    void BuildingLayout<Scalar>::drawTerrain(Color color, float colorAlpha, bool wireframe, float wireframeAlpha) const
    {
        if (model.meshCount == 0)
            return;
        DrawModel(model, {0, 0, 0}, 1.0f, Fade(color, colorAlpha));
        if (wireframe)
        {
            // 绘制线框
            DrawModelWires(model, {0, 0, 0}, 1.0f, Fade(RL_BLACK, wireframeAlpha));
        }
    }

    struct RVDModel : torch::nn::Module
    {
        // ===== Fixed data =====
        torch::Tensor grid_xy;   // [G, 2]
        torch::Tensor terrain_h; // [G]
        torch::Tensor site_xy;   // [N, 2]
        torch::Tensor weights;
        float beta;

        // ===== Trainable =====
        torch::Tensor h_cell; // [N]

        RVDModel(
            const torch::Tensor &grid_xy_,
            const torch::Tensor &terrain_h_,
            const torch::Tensor &site_xy_,
            float cell_area_,
            float beta_ = 10.f)
            : grid_xy(grid_xy_),
              terrain_h(terrain_h_),
              site_xy(site_xy_),
              beta(beta_)
        {
            const int N = site_xy.size(0);

            h_cell = register_parameter(
                "h_cell",
                torch::zeros({N}, torch::requires_grad()));
            // grid_xy: [G, 2]
            // site_xy: [N, 2]

            // ===== L∞ distance =====
            // |x - p| : [G, N, 2]
            auto diff = grid_xy.unsqueeze(1) - site_xy.unsqueeze(0);
            auto dist = torch::amax(torch::abs(diff), /*dim=*/2); // [G, N]

            // ===== Soft Voronoi =====
            weights = torch::softmax(-beta * dist, /*dim=*/1); // [G, N]

            // auto w_cpu = weights.detach().to(torch::kCPU);

            // for (int g = 0; g < grid_xy.size(0); ++g)
            // {
            //     std::cout << "Grid " << g << " weights: ";
            //     for (int i = 0; i < site_xy.size(0); ++i)
            //     {
            //         std::cout << w_cpu[g][i].item<float>() << " ";
            //     }
            //     std::cout << std::endl;
            // }
        }

        torch::Tensor forward()
        {

            // ===== Target height =====
            auto h_target = torch::matmul(weights, h_cell); // [G]

            // ===== Local earthwork =====
            auto diff_h = h_target - terrain_h;
            auto E_local = torch::sum(diff_h * diff_h);

            // ===== Global balance =====
            auto V = torch::sum(diff_h);
            auto E_balance = V * V;

            // ===== Total =====
            return E_local + 10.0f * E_balance;
        }

        void drawGrids(float z = 0.f, float size = 1.f) const
        {
            int G = grid_xy.size(0);
            int N = site_xy.size(0);

            // 1. 预先生成每个 site 的“cell 颜色”
            std::vector<Color> cellColors(N);
            for (int i = 0; i < N; ++i)
            {
                float hue = float(i) / float(N);
                cellColors[i] = renderUtil::ColorFromHue(hue);
            }

            // 2. 把 weights 拉到 CPU（只做可视化）
            auto w_cpu = weights.detach().to(torch::kCPU);

            for (int g = 0; g < G; ++g)
            {
                // ---- grid position ----
                float x = grid_xy[g][0].item<float>();
                float y = grid_xy[g][1].item<float>();

                // ---- collect weights ----
                std::vector<float> w(N);
                for (int i = 0; i < N; ++i)
                    w[i] = w_cpu[g][i].item<float>();

                // ---- mixed color ----
                Color c = renderUtil::mixColor(cellColors, w);

                // ---- draw ----
                DrawCube(
                    Vector3{x, z, -y},
                    size, size, size,
                    c);
            }
        }
    };

    struct SoftRVDShowData
    {
        torch::Tensor grid_xy; // [G,2]
        torch::Tensor weights; // [G,N]
        torch::Tensor floors;  // [N]
        torch::Tensor dh;      // [N]
        torch::Tensor base_h;  // scalar

        int G = 0;
        int N = 0;

        // =========================
        // DRAW BUILDING MASS
        // =========================
        void draw(
            float floor_height = 4.f,
            float grid_size = 1.f,
            Eigen::Vector2f offset = Eigen::Vector2f(0.f, 0.f)) const;
    };

    class SoftRVDModel : public torch::nn::Module
    {
    public:
        // ===================== Fixed inputs =====================
        torch::Tensor grid_xy;   // [G,2]
        torch::Tensor terrain_h; // [G]
        torch::Tensor site_xy;   // [N,2]

        int G = 0;
        int N = 0;

        float k;
        float tau;
        float far;

        std::vector<int> courtyard_ids;

        // ===================== Soft RVD =====================
        torch::Tensor weights; // [G,N] 固定 soft-RVD 权重

        // ===================== Trainable =====================
        torch::Tensor floor_logits; // [N,5] -> floors ∈ {0..4}
        torch::Tensor delta_logits; // [N,5] -> Δh ∈ {0,2,4,6,8}

        // ===================== Derived =====================
        torch::Tensor base_height; // scalar (最低地基)

        // ===================== Hyper =====================
        float lambda_far = 1.0f;
        float lambda_terrain = 0.5f;
        float lambda_entropy = 0.01f;
std::unique_ptr<torch::optim::Adam> lloyd_optimizer;
    public:
        SoftRVDModel(
            const torch::Tensor &grid_xy_,
            const torch::Tensor &terrain_h_,
            const torch::Tensor &site_xy_,
            float far_,
            const std::vector<int> &courtyard_ids_,
            float k_,
            float tau_);
        void computeWeights();
        torch::Tensor forward();
        torch::Tensor energyLloyd();
        void optimize(SoftRVDShowData &showData,
                      int iters = 300,
                      float lr = 0.05f,
                      int verbose_every = 50);
        void optimizeLloyd(int iters = 250,
                           float lr = 0.05f,
                           int verbose_every = 50);
        void stepOptimizeLloyd(int& curIter,const int maxIter);
        void stepOptimize(int& curIter, const int maxIter);
        void drawGrids(float z = 0.f, float size = 1.f, const Eigen::Vector2f &offset = Eigen::Vector2f(0.f, 0.f)) const;
        void drawTerrain(const std::vector<float> &heights, float z = 0.f, float size = 1.f, const Eigen::Vector2f &offset = Eigen::Vector2f(0.f, 0.f)) const;
    };

}