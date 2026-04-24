#pragma once

#include "terrain.h"
#include "geo.h"
#include "diffVoronoi.h"
#include <torch/torch.h>
#include "grid.h"
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
    RectCell<Scalar> RectVoronoi2D<Scalar>::computeCell(int idx, const SeparateRegion &region) const
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
        // using SurfaceMesh = geo::SurfaceMesh;
        template <typename Scalar>
        using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
        using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

    public:
        BuildingLayout() = default;
        BuildingLayout(const Polyline2_t<Scalar> &site_, const terrain::Terrain &terrain);
        void upload() { model = LoadModelFromMesh(geo::buildRaylibMesh(meshData)); }
        void drawTerrain(Color color, float colorAlpha, bool wireframe, float wireframeAlpha, Eigen::Vector3f position = {0.f, 0.f, 0.f}) const;

        Polyline2_t<Scalar> toWorldFromRotedSite(const Polyline2_t<Scalar> &rotedSite) const;
        inline Vector2<Scalar> toWorldPt(const Vector2<Scalar> &p) const { return Rinv * (p - center) + center; }
        std::vector<Vector2<Scalar>> toWorldPts(const std::vector<Vector2<Scalar>> &pts) const;

    private:
        void initLayout(const terrain::Terrain &terrain);

    private:
        // CPU mesh
        // geo::MeshData orthogonalMesh;
        Model model;
        // GPU mesh
    public:
        Polyline2_t<Scalar> site;
        Polyline2_t<Scalar> oriRect; // 原始矩形（未旋转）
        Vector2<Scalar> center;
        Polyline2_t<Scalar> rotedRect; // 旋转后的矩形
        Polyline2_t<Scalar> rotedSite; // 旋转后的地块
        Polyline2_t<Scalar> realSite;  // 旋转回去的地块
        Eigen::AlignedBox<Scalar, 2> rotedBound;
        Eigen::Matrix<Scalar, 2, 2> Rinv;
        grid::InverseTran<Scalar> inverseTran;
        std::vector<Scalar> heightMap;
        std::vector<Vector2<Scalar>> rotedCenters;
        std::vector<Eigen::Vector3<Scalar>> sampleCenters;
        Scalar divGap;
        // std::vector<Vector2<Scalar>> rotedVertices;
        std::vector<Eigen::Vector3<Scalar>> grids;
        geo::MeshData meshData;
        // SurfaceMesh heMesh;
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
        geo::OBB2<Scalar> obb(site.points);
        Eigen::Matrix<Scalar, 2, 2> R = geo::rotationToXAxis(obb.axis0);
        Rinv = R.transpose();

        Polyline2_t<Scalar> rotPoly = geo::rotatePoly(site, R);
        rotedRect = geo::getMaxRectInPolyWithRatio(rotPoly, 0.5, 1.8);

        // =========================
        // rotate back to world
        // =========================
        std::vector<Eigen::Vector2<Scalar>> originalPts;
        for (const auto &p : rotedRect.points)
            originalPts.emplace_back(Rinv * p);

        oriRect = Polyline2_t<Scalar>(originalPts, true);

      
        heightMap.clear();
        meshData.vertices.clear();
        meshData.indices.clear();
        rotedCenters.clear();
        grids.clear();
        sampleCenters.clear();

        // =========================
        // base rect
        // =========================
        Vector2<Scalar> p0 = oriRect.points[0]; // 世界坐标下原始矩形顶点0
        Vector2<Scalar> p1 = oriRect.points[1]; // 世界坐标下原始矩形顶点1
        Vector2<Scalar> p3 = oriRect.points[3]; // 世界坐标下原始矩形顶点3

        Vector2<Scalar> edgeW = p1 - p0; // 世界坐标下宽边的方向
        Vector2<Scalar> edgeH = p3 - p0; // 世界坐标下长边的方向

        Scalar width = edgeW.norm(); // 宽的长度
        Scalar height = edgeH.norm(); //长边的长度

        Scalar newW = std::floor(width); // 向下取整
        Scalar newH = std::floor(height); //向下取整

        center = (p0 + oriRect.points[2]) * Scalar(0.5); // 中心点

        Vector2<Scalar> dirW = edgeW.normalized(); // 世界坐标宽的方向
        Vector2<Scalar> dirH = edgeH.normalized(); // 世界坐标长的方向

        Scalar hw = newW * Scalar(0.5); //一半的宽
        Scalar hh = newH * Scalar(0.5); //一半的长

        Scalar minW = std::min(newW, newH); //真正的宽
        Scalar maxW = std::max(newW, newH); //真正的长

        divGap = std::min(Scalar(0.3 * minW), Scalar(0.2 * maxW));

        Vector2<Scalar> newP0 = center - dirW * hw - dirH * hh; // 边长取整后，世界坐标下的新端点0
        Vector2<Scalar> newP1 = center + dirW * hw - dirH * hh; // 边长取整后，世界坐标下的新端点1
        Vector2<Scalar> newP2 = center + dirW * hw + dirH * hh; // 边长取整后，世界坐标下的新端点2
        Vector2<Scalar> newP3 = center - dirW * hw + dirH * hh; // 边长取整后，世界坐标下的新端点3

        int nx = static_cast<int>(newW); //整数值宽
        int ny = static_cast<int>(newH); //整数值长

    

        // =========================
        // vertices
        // =========================
        for (int j = 0; j <= ny; ++j)
        {
            for (int i = 0; i <= nx; ++i)
            {
                //世界坐标下的网格顶点
                Vector2<Scalar> v = newP0 + dirW * Scalar(i) + dirH * Scalar(j);

                Scalar h = Scalar(0);
                terrain.sampleHeightAt(h, v);
                //世界坐标下的网格顶点
                grids.push_back({v.x(), v.y(), h});

                //局部坐标系下的顶点
                Eigen::Vector2<Scalar> rv = R * v;

                geo::Vertex mv({(float)rv.x(),
                                (float)rv.y(),
                                (float)h});
                //局部坐标系下的mesh
                meshData.vertices.push_back(mv);
            }
        }

        // =========================
        // faces + centers
        // =========================
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                uint32_t i0 = j * (nx + 1) + i;
                uint32_t i1 = j * (nx + 1) + (i + 1);
                uint32_t i2 = (j + 1) * (nx + 1) + (i + 1);
                uint32_t i3 = (j + 1) * (nx + 1) + i;

                // ===== mesh =====
                meshData.indices.push_back(i0);
                meshData.indices.push_back(i1);
                meshData.indices.push_back(i2);

                meshData.indices.push_back(i0);
                meshData.indices.push_back(i2);
                meshData.indices.push_back(i3);

                // ============================================
                // ⭐ 正确 center：直接用 vertex 平均
                // ============================================
                const auto &v0 = meshData.vertices[i0].position;
                const auto &v1 = meshData.vertices[i1].position;
                const auto &v2 = meshData.vertices[i2].position;
                const auto &v3 = meshData.vertices[i3].position;

                Eigen::Vector2f center2D(
                    (v0.x() + v1.x() + v2.x() + v3.x()) * 0.25f,
                    (v0.y() + v1.y() + v2.y() + v3.y()) * 0.25f);

                //center2D也是局部坐标系下的中心点
                rotedCenters.push_back(center2D);

                // ============================================
                // ⭐ 反算回 world（给 height / sample 用）
                // ============================================
                Eigen::Vector2f world2D = Rinv*center2D;

                Scalar h = Scalar(0);
                terrain.sampleHeightAt(h, Vector2<Scalar>(world2D.x(), world2D.y()));
                //局部坐标系的mesh的网格面face的中心点，返回到实际世界坐标系下的world2D坐标，去采样
                heightMap.push_back(h);
                sampleCenters.push_back({world2D.x(), world2D.y(), h});
            }
        }

        // auto getV = [&](int j, int i) -> Eigen::Vector2f
        // {
        //     int idx = j * (nx + 1) + i;
        //     const auto &v = meshData.vertices[idx].position;
        //     return Eigen::Vector2f(v.x(), v.y());
        // };

        // std::vector<Vector2<Scalar>> rectPts = {
        //     getV(0, 0),
        //     getV(0, nx),
        //     getV(ny, nx),
        //     getV(ny, 0)};

        std::vector<Vector2<Scalar>> realSitePts = {newP0, newP1, newP2, newP3, newP0};
        //世界坐标系下规整了尺寸的矩形site
        realSite = Polyline2_t<Scalar>(realSitePts, true);
        //局部坐标系下规整了尺寸的矩形site
        rotedSite = geo::rotatePoly(realSite,R);
        geo::computeVertexNormals(meshData);
        inverseTran = grid::InverseTran<Scalar>(center, Rinv);
        // std::cout << "center: " << inverseTran.center.transpose() << std::endl;
        // std::cout << "Rinv: " << std::endl
        //           << inverseTran.Rinv << std::endl;
      
    }

    template <typename Scalar>
    geo::Polyline2_t<Scalar> BuildingLayout<Scalar>::toWorldFromRotedSite(const Polyline2_t<Scalar> &rotedSite) const
    {
        std::vector<Eigen::Vector2<Scalar>> worldPts;
        worldPts.reserve(rotedSite.points.size());

        for (const auto &p : rotedSite.points)
        {
            Eigen::Vector2<Scalar> wp = Rinv * (p - center) + center;
            worldPts.push_back(wp);
        }
        return geo::Polyline2_t<Scalar>(worldPts, true);
    }

    template <typename Scalar>
    std::vector<Eigen::Vector2<Scalar>> BuildingLayout<Scalar>::toWorldPts(const std::vector<Eigen::Vector2<Scalar>> &pts) const
    {
        std::vector<Eigen::Vector2<Scalar>> worldpts;
        for (const auto &p : pts)
            worldpts.push_back(toWorldPt(p));

        return worldpts;
    }

    template <typename Scalar>
    void BuildingLayout<Scalar>::drawTerrain(Color color, float colorAlpha, bool wireframe, float wireframeAlpha, Eigen::Vector3f position) const
    {
        if (model.meshCount == 0)
            return;
        DrawModel(model, {position.x(), position.z(), -position.y()}, 1.0f, Fade(color, colorAlpha));
        if (wireframe)
        {
            // 绘制线框
            DrawModelWires(model, {position.x(), position.z(), -position.y()}, 1.0f, Fade(RL_BLACK, wireframeAlpha));
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
        torch::Tensor H_site;
        std::vector<int> courtyard_ids;
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

    struct SoftRVDOutput
    {
        torch::Tensor H_site;
        torch::Tensor floors;
        torch::Tensor dh;
        torch::Tensor weights;
        torch::Tensor base_h;
    };

    class SoftRVDModel : public torch::nn::Module
    {
    public:
        // ===================== Fixed inputs =====================
        torch::Tensor grid_xy;   // [G,2]
        torch::Tensor terrain_h; // [G]
        torch::Tensor site_xy;   // [N,2]
        torch::Tensor global_offset;
        mutable SoftRVDOutput lastOutput;
        int G = 0;
        int N = 0;
        int iter = 0;
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
        std::vector<int> isAffectLands;
        // ===================== Hyper =====================
        // float lambda_far = 1.0f;
        // float lambda_terrain = 0.5f;
        float lambda_entropy = 0.05f;
        std::unique_ptr<torch::optim::Adam> lloyd_optimizer;
        std::unique_ptr<torch::optim::Adam> optimizer;

    public:
        SoftRVDModel() = default;
        SoftRVDModel(
            const torch::Tensor &grid_xy_,
            const torch::Tensor &terrain_h_,
            const torch::Tensor &site_xy_,
            float far_,
            const std::vector<int> &courtyard_ids_,
            const std::vector<int> &isAffectLands_, // ⭐ 新增
            float k_,
            float tau_)
            : grid_xy(grid_xy_),
              terrain_h(terrain_h_),
              site_xy(site_xy_),
              far(far_),
              courtyard_ids(courtyard_ids_),
              k(k_),
              tau(tau_)
        {
            G = grid_xy.size(0);
            N = site_xy.size(0);

            site_xy = register_parameter(
                "site_xy",
                site_xy_.clone());

            auto device = grid_xy.device();

            // ===============================
            // ⭐ 处理 isAffectLands 长度
            // ===============================
            isAffectLands = isAffectLands_; // 先拷贝

            if ((int)isAffectLands.size() < N)
            {
                // 不够 → 补 1（默认影响）
                isAffectLands.resize(N, 1);
            }
            else if ((int)isAffectLands.size() > N)
            {
                // 超出 → 截断
                isAffectLands.resize(N);
            }

            // ⭐⭐⭐ 楼层限制：0~3（4种）
            floor_logits = register_parameter(
                "floor_logits",
                0.01f * torch::randn({N, 4}, device)); // ⭐ 改这里

            delta_logits = register_parameter(
                "delta_logits",
                0.01f * torch::randn({N, 5}, device));

            global_offset = register_parameter(
                "global_offset",
                torch::zeros({1}, device));

            computeWeights();
            register_buffer("weights", weights);

            auto A = weights.sum(0);
            auto h_site_mean =
                (weights.transpose(0, 1).matmul(terrain_h)) / (A + 1e-6);

            base_height = h_site_mean.min().detach();

            lloyd_optimizer = std::make_unique<torch::optim::Adam>(
                this->parameters(),
                torch::optim::AdamOptions(0.05));
            optimizer = std::make_unique<torch::optim::Adam>(
                this->parameters(),
                torch::optim::AdamOptions(0.05));
        }

        void computeWeights();
        std::pair<torch::Tensor, SoftRVDOutput> forward();
        torch::Tensor energyLloyd();
        void optimize(/* SoftRVDShowData &showData,  */ int maxIter = 120);
        void optimizeLloyd(int iters = 250,
                           float lr = 0.05f,
                           int verbose_every = 50);
        void stepOptimizeLloyd(int &curIter, const int maxIter);
        void stepOptimize(SoftRVDShowData &showData, int &curIter, int maxIter, bool &isOptimizing);
        void drawGrids(float z = 0.f, float size = 1.f, const Eigen::Vector2f &offset = Eigen::Vector2f(0.f, 0.f)) const;
        void drawTerrain(const std::vector<float> &heights, float z = 0.f, float size = 1.f, const Eigen::Vector2f &offset = Eigen::Vector2f(0.f, 0.f)) const;
        std::pair<grid::CellRegion, grid::FloorSystem> buildCellRegion(const grid::CellGenerator &cellGen,
                                                                       const geo::MeshData &originalMesh, const grid::InverseTran<float> &tran = grid::InverseTran<float>()) const;
    };

}