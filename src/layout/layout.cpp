#include "layout.h"

namespace layout
{

    void SoftRVDModel::computeWeights()
    {
        auto device = grid_xy.device();
        weights = torch::zeros({G, N}, device);

        auto gx = grid_xy.index({torch::indexing::Slice(), 0});
        auto gy = grid_xy.index({torch::indexing::Slice(), 1});

        for (int i = 0; i < N; ++i)
        {
            auto logw = torch::zeros({G}, device);
            auto px = site_xy[i][0];
            auto py = site_xy[i][1];

            for (int q = 0; q < N; ++q)
            {
                if (q == i)
                    continue;

                auto qx = site_xy[q][0];
                auto qy = site_xy[q][1];

                auto dx = px - qx;
                auto dy = py - qy;

                auto scores = torch::stack({dx - torch::abs(dy),
                                            -dx - torch::abs(dy),
                                            dy - torch::abs(dx),
                                            -dy - torch::abs(dx)});

                auto p = torch::softmax(tau * scores, 0);

                auto mx = 0.5f * (px + qx);
                auto my = 0.5f * (py + qy);

                logw += p[0] * torch::nn::functional::logsigmoid(k * (gx - mx));
                logw += p[1] * torch::nn::functional::logsigmoid(k * (mx - gx));
                logw += p[2] * torch::nn::functional::logsigmoid(k * (gy - my));
                logw += p[3] * torch::nn::functional::logsigmoid(k * (my - gy));
            }

            weights.index_put_({torch::indexing::Slice(), i}, torch::exp(logw));
        }

        auto max_w = std::get<0>(weights.max(1, true));
        weights = weights + 0.05f * (weights == max_w).to(weights.dtype());
        weights = weights / (weights.sum(1, true) + 1e-6);
    }

    std::pair<torch::Tensor, SoftRVDOutput> SoftRVDModel::forward()
    {
        auto device = grid_xy.device();

        computeWeights();

        auto A = weights.sum(0);

        auto h_site_mean =
            (weights.transpose(0, 1).matmul(terrain_h)) / (A + 1e-6);

        // ===============================
        // robust max
        // ===============================
        float k_max = 50.0f;

        auto log_w = torch::log(weights + 1e-6);
        auto val = k_max * (terrain_h.unsqueeze(1) + log_w);

        auto h_site_max =
            torch::logsumexp(val, 0) / k_max;

        auto h_base = h_site_mean.min().detach();

        // ===============================
        // floors（1~3）
        // ===============================
        auto floor_vals = torch::tensor({1.f, 2.f, 3.f}, device);

        auto p_floor = torch::softmax(
            floor_logits.index({torch::indexing::Slice(),
                                torch::indexing::Slice(0, 3)}),
            1);

        auto floors = (p_floor * floor_vals).sum(1);

        // ===============================
        // courtyard
        // ===============================
        auto courtyard_mask = torch::zeros({N}, device);

        if (!courtyard_ids.empty())
        {
            auto ids = torch::tensor(courtyard_ids, torch::kLong).to(device);
            courtyard_mask.index_put_({ids}, 1.0f);
            floors = floors * (1.0f - courtyard_mask);
        }

        auto building_mask = (1.0f - courtyard_mask); // ⭐ 用于shape过滤

        // ===============================
        // dh
        // ===============================
        auto delta_vals = torch::tensor({0.f, 2.f, 4.f, 6.f, 8.f}, device);
        auto dh = (torch::softmax(delta_logits, 1) * delta_vals).sum(1);

        // ===============================
        // ground / float
        // ===============================
        auto affect_mask = torch::tensor(isAffectLands, torch::kFloat32).to(device);

        auto H_ground = h_base + global_offset + dh;

        float clearance = 1.0f;
        auto H_float = h_site_max + clearance;

        auto H_site =affect_mask * H_ground + (1.0f - affect_mask) * H_float;

        SoftRVDOutput out;
        out.H_site = H_site;
        out.floors = floors;
        out.dh = dh;
        out.weights = weights;
        out.base_h = h_base;

        lastOutput = out;
        // ===============================
        // CUT / FILL
        // ===============================
        auto H_grid = weights.matmul(H_site);

        auto cut = torch::relu(H_grid - terrain_h);
        auto fill = torch::relu(terrain_h - H_grid);

        auto L_balance =
            (cut.sum() - fill.sum()).pow(2) / (G * G);

        // ===============================
        // adjacency
        // ===============================
        auto affinity = weights.transpose(0, 1).matmul(weights);

        auto adj = torch::relu(affinity - 0.05f);

        // ===============================
        // roof constraint ⭐
        // ===============================
        float floor_h = 3.0f;

        auto roof = H_site + floors * floor_h;

        auto Hi = H_site.unsqueeze(1);
        auto Hj = H_site.unsqueeze(0);

        auto Ri = roof.unsqueeze(1);
        auto Rj = roof.unsqueeze(0);

        auto violation1 = torch::relu(Hi - (Rj + 2.0f));
        auto violation2 = torch::relu(Hj - (Ri + 2.0f));

        auto L_roof =
            (adj * (violation1.pow(2) + violation2.pow(2))).mean();

        // ===============================
        // Lloyd
        // ===============================
        auto centroid =(weights.transpose(0, 1).matmul(grid_xy)) / (A.unsqueeze(1) + 1e-6);

        auto L_lloyd = (site_xy - centroid).pow(2).sum(1).mean();

        // ===============================
        // ⭐⭐⭐ AABB SHAPE（核心替换）
        // ===============================
        float k_box = 20.0f;

        auto x = grid_xy.select(1, 0).unsqueeze(1);
        auto y = grid_xy.select(1, 1).unsqueeze(1);

        auto x_max = torch::logsumexp(k_box * (x + log_w), 0) / k_box;

        auto x_min = -torch::logsumexp(k_box * (-x + log_w), 0) / k_box;

        auto y_max = torch::logsumexp(k_box * (y + log_w), 0) / k_box;

        auto y_min = -torch::logsumexp(k_box * (-y + log_w), 0) / k_box;

        auto width = x_max - x_min;
        auto height = y_max - y_min;

        auto ratio = torch::max(width / (height + 1e-6),
                                height / (width + 1e-6));

        // ⭐ shape loss（只作用建筑）
        auto L_shape =
            (torch::relu(ratio - 2.0f).pow(2) * building_mask).sum() /
            (building_mask.sum() + 1e-6);

        // ===============================
        // FAR
        // ===============================
        auto total_floors = (A * floors * building_mask).sum();

        auto L_far = ((total_floors - G * far) / G).pow(2);

        float actual_far = total_floors.detach().item<float>() / float(G);

        // ===============================
        // FINAL LOSS
        // ===============================
        auto loss = 1.0f * L_far + 0.5f * L_balance + 1.0f * L_roof + 0.3f * L_lloyd + 0.5f * L_shape; // ⭐ 提高权重（关键）

        // ===============================
        // DEBUG
        // ===============================
        static int iter = 0;

        if (iter % 10 == 0)
        {
            float loss_val = loss.item<float>();
            float far_val = L_far.item<float>();
            float balance_val = L_balance.item<float>();
            float roof_val = L_roof.item<float>();
            float lloyd_val = L_lloyd.item<float>();
            float shape_val = L_shape.item<float>();

            // ⭐ 平均 aspect ratio（方便观察整体形状）
            float ratio_mean =
                ratio.mean().item<float>();

            // ⭐ 最大 ratio（专门抓细长异常）
            float ratio_max =
                ratio.max().item<float>();

            std::cout << "Iter " << iter
                      << " - Loss: " << loss_val
                      << " (far: " << far_val
                      << ", balance: " << balance_val
                      << ", roof: " << roof_val
                      << ", lloyd: " << lloyd_val
                      << ", shape: " << shape_val
                      << ", ratio_mean: " << ratio_mean
                      << ", ratio_max: " << ratio_max
                      << ")\n";
        }

        iter++;

        return {loss, out};
    }

    grid::CellRegion SoftRVDModel::buildCellRegion(
        const grid::CellGenerator &cellGen) const
    {
        const auto &out = lastOutput;

        auto w_cpu = weights.detach().to(torch::kCPU);
        auto H_cpu = out.H_site.detach().to(torch::kCPU);
        auto floor_cpu = out.floors.detach().to(torch::kCPU);

        // =========================
        // 1. assignment
        // =========================
        std::vector<std::vector<int>> groupIndices(N);
        std::vector<int> gridOwner(G);

        for (int g = 0; g < G; ++g)
        {
            int best_i = 0;
            float max_w = -1e9f;

            for (int i = 0; i < N; ++i)
            {
                float w = w_cpu[g][i].item<float>();
                if (w > max_w)
                {
                    max_w = w;
                    best_i = i;
                }
            }

            groupIndices[best_i].push_back(g);
            gridOwner[g] = best_i;
        }

        // =========================
        // 2. base height
        // =========================
        std::vector<float> baseHeights(N);
        for (int i = 0; i < N; ++i)
            baseHeights[i] = H_cpu[i].item<float>();

        // =========================
        // 3. floors
        // =========================
        std::vector<int> floors(N);
        for (int i = 0; i < N; ++i)
            floors[i] = (int)std::round(floor_cpu[i].item<float>());

        // =========================
        // ⭐ DEBUG PRINT
        // =========================
        std::cout << "\n================ CELL REGION DEBUG ================\n";

        for (int i = 0; i < N; ++i)
        {
            std::cout << "Site " << i
                      << " | baseH=" << baseHeights[i]
                      << " | floors=" << floors[i]
                      << " | grids=[";

            const auto &vec = groupIndices[i];

            for (size_t k = 0; k < vec.size(); ++k)
            {
                std::cout << vec[k];
                if (k + 1 < vec.size())
                    std::cout << ",";
            }

            std::cout << "]\n";
        }

        std::cout << "==================================================\n\n";

        // =========================
        // 4. return
        // =========================
        return grid::CellRegion(
            3,
            &cellGen.cells,
            groupIndices,
            baseHeights,
            floors);
    }

    void SoftRVDModel::drawGrids(float z, float size, const Eigen::Vector2f &offset) const
    {

        // 1. 预先生成每个 site 的“cell 颜色”
        std::vector<Color> cellColors(N);
        for (int i = 0; i < N; ++i)
        {
            float hue = float(i) / float(N);
            cellColors[i] = renderUtil::ColorFromHue(hue);
            float sx = site_xy[i][0].item<float>();
            float sy = site_xy[i][1].item<float>();

            // 画一个小 cube / sphere 表示 site
            DrawCube(
                Vector3{sx + offset.x(), 0, -(sy + offset.y())},
                size,
                size,
                size,
                RL_BLACK);
        }

        // 2. 把 weights 拉到 CPU（只做可视化）
        auto w_cpu = weights.detach().to(torch::kCPU);
        auto h_cpu = weights.detach().to(torch::kCPU);
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
            DrawCube(Vector3{x + offset.x(), 0, -(y + offset.y())}, size, 0, size, Fade(c, 0.3f));
            DrawCubeWires(Vector3{x + offset.x(), 0, -(y + offset.y())}, size, 0, size, RL_BLACK);
        }
    }

    void SoftRVDModel::drawTerrain(const std::vector<float> &heights, float z, float size, const Eigen::Vector2f &offset) const
    {

        // 1. 预先生成每个 site 的“cell 颜色”
        std::vector<Color> cellColors(N);
        for (int i = 0; i < N; ++i)
        {
            float hue = float(i) / float(N);
            cellColors[i] = renderUtil::ColorFromHue(hue);
        }

        // 2. 把 weights 拉到 CPU（只做可视化）
        auto w_cpu = weights.detach().to(torch::kCPU);
        auto h_cpu = weights.detach().to(torch::kCPU);
        for (int g = 0; g < G; ++g)
        {
            // ---- grid position ----
            float x = grid_xy[g][0].item<float>();
            float y = grid_xy[g][1].item<float>();

            // ---- collect weights ----
            // std::vector<float> w(N);
            // for (int i = 0; i < N; ++i)
            //     w[i] = w_cpu[g][i].item<float>();

            // // ---- mixed color ----
            // Color c = renderUtil::mixColor(cellColors, w);
            // float h = h_cpu[g].item<float>();
            // ---- draw ----
            DrawCube(Vector3{x + offset.x(), heights[g], -(y + offset.y())}, size, 0, size, RL_GRAY);
            // DrawCubeWires(Vector3{x + offset.x(), heights[g], -(y + offset.y())},size, size, size,RL_BLACK);
        }
    }
    torch::Tensor SoftRVDModel::energyLloyd()
    {
        computeWeights();
        auto gx = grid_xy.unsqueeze(1);
        auto sx = site_xy.unsqueeze(0);

        auto dist2 = (gx - sx).pow(2).sum(-1);
        auto W = weights.detach();
        auto E = (W * dist2).sum();

        return E;
    }

    void SoftRVDModel::optimizeLloyd(
        int iters,
        float lr,
        int verbose_every)
    {
        torch::optim::Adam optimizer(
            this->parameters(),
            torch::optim::AdamOptions(lr));

        for (int iter = 0; iter < iters; ++iter)
        {
            optimizer.zero_grad();

            auto loss = energyLloyd(); // ← 改这里

            loss.backward();
            optimizer.step();

            if (iter % verbose_every == 0 || iter == iters - 1)
            {
                std::cout
                    << "[Lloyd Iter " << iter
                    << "] Energy = "
                    << loss.item<float>()
                    << std::endl;
            }
        }
    }

    void SoftRVDModel::stepOptimizeLloyd(int &curIter, const int maxIter)
    {
        if (curIter > maxIter)
            return;
        lloyd_optimizer->zero_grad();
        auto loss = energyLloyd();
        loss.backward();
        lloyd_optimizer->step();
        curIter++;
        if (curIter % 50 == 0 || curIter == maxIter - 1)
        {
            std::cout
                << "[Lloyd Iter " << curIter
                << "] Energy = "
                << loss.item<float>()
                << std::endl;
        }
    }

    void SoftRVDModel::stepOptimize(
        SoftRVDShowData &showData,
        int &curIter,
        int maxIter,
        bool &isOptimizing)
    {
        if (curIter >= maxIter)
        {
            isOptimizing = false;
            return;
        }

        optimizer->zero_grad();

        float t = float(curIter) / float(std::max(maxIter - 1, 1));
        float s = t * t * (3.f - 2.f * t);

        lambda_entropy = 0.05f * (1.0f - s);

        // ⭐ forward
        auto [loss, out] = forward();
        loss.backward();
        optimizer->step();

        showData.grid_xy = grid_xy.detach().clone();
        showData.weights = out.weights.detach().clone();
        showData.floors = out.floors.detach().clone();
        showData.dh = out.dh.detach().clone();
        showData.base_h = out.base_h.detach().clone();
        showData.H_site = out.H_site.detach().clone();

        showData.G = G;
        showData.N = N;

        curIter++;
    }

    void SoftRVDModel::optimize(
        SoftRVDShowData &showData,
        int iters,
        float lr,
        int verbose_every)
    {
        // torch::optim::Adam optimizer(
        //     this->parameters(),
        //     torch::optim::AdamOptions(lr));

        // for (int iter = 0; iter < iters; ++iter)
        // {
        //     optimizer.zero_grad();

        //     float t = float(iter) / float(std::max(iters - 1, 1));
        //     float s = t * t * (3.f - 2.f * t);

        //     lambda_entropy = 0.05f * (1.0f - s);

        //     auto loss = forward();

        //     // ⭐⭐⭐ 防止 backward 崩
        //     if (!loss.requires_grad())
        //     {
        //         std::cout << "❌ loss has no grad!" << std::endl;
        //         return;
        //     }

        //     loss.backward();
        //     optimizer.step();

        //     // =========================
        //     // UPDATE SHOW DATA
        //     // =========================
        //     {
        //         auto device = grid_xy.device();

        //         auto floor_vals = torch::arange(0, 5, device);
        //         auto p_floor = torch::softmax(floor_logits, 1);
        //         auto floors_now = (p_floor * floor_vals).sum(1);

        //         // 同样要 mask（保持一致）
        //         if (!courtyard_ids.empty())
        //         {
        //             auto mask = torch::ones_like(floors_now);
        //             auto ids_tensor = torch::tensor(courtyard_ids, torch::dtype(torch::kLong).device(device));
        //             mask.index_put_({ids_tensor}, 0.0f);
        //             floors_now = floors_now * mask;
        //         }

        //         auto delta_vals =
        //             torch::tensor({0.f, 2.f, 4.f, 6.f, 8.f}, device);

        //         auto p_delta = torch::softmax(delta_logits, 1);
        //         auto dh_now = (p_delta * delta_vals).sum(1);

        //         dh_now = dh_now * torch::relu(floors_now - 1.0f);

        //         showData.grid_xy = grid_xy;
        //         showData.weights = weights;
        //         showData.floors = floors_now;
        //         showData.dh = dh_now;
        //         showData.base_h = base_height;
        //         showData.G = G;
        //         showData.N = N;
        //         showData.courtyard_ids = courtyard_ids;
        //     }

        //     if (iter % verbose_every == 0 || iter == iters - 1)
        //     {
        //         std::cout
        //             << "[Iter " << iter
        //             << "] loss=" << loss.item<float>()
        //             << std::endl;
        //     }
        // }
    }

    void SoftRVDShowData::draw(float floor_height, float grid_size, Eigen::Vector2f offset) const
    {
        // std::cout << "[DRAW DEBUG]"
        //           << " grid_xy defined=" << grid_xy.defined()
        //           << " size=" << (grid_xy.defined() ? grid_xy.sizes() : torch::IntArrayRef{})
        //           << " weights defined=" << weights.defined()
        //           << " size=" << (weights.defined() ? weights.sizes() : torch::IntArrayRef{})
        //           << " floors defined=" << floors.defined()
        //           << std::endl;
        if (!grid_xy.defined() || !weights.defined() || !floors.defined())
            return;

        auto grid_cpu = grid_xy.detach().to(torch::kCPU);
        auto w_cpu = weights.detach().to(torch::kCPU);
        auto f_cpu = floors.detach().to(torch::kCPU);
        auto hsite_cpu = H_site.detach().to(torch::kCPU);

        std::vector<Color> cellColors(N);
        for (int i = 0; i < N; ++i)
        {
            float hue = float(i) / float(N);
            cellColors[i] = renderUtil::ColorFromHue(hue);
        }

        for (int g = 0; g < G; ++g)
        {
            float gx = grid_cpu[g][0].item<float>();
            float gy = grid_cpu[g][1].item<float>();

            int best_i = 0;
            float max_w = -1e9f; // ⭐ 修复

            for (int i = 0; i < N; ++i)
            {
                float w = w_cpu[g][i].item<float>();
                if (w > max_w)
                {
                    max_w = w;
                    best_i = i;
                }
            }

            float floors_soft = f_cpu[best_i].item<float>();
            int floors_int = (int)std::round(floors_soft);

            for (int cid : courtyard_ids)
            {
                if (best_i == cid)
                {
                    floors_int = 0;
                    break;
                }
            }

            if (floors_int <= 0)
                continue;

            float height = floors_int * floor_height;
            float z_base = hsite_cpu[best_i].item<float>();

            DrawCube(
                Vector3{
                    gx + offset.x(),
                    z_base + height * 0.5f,
                    -(gy + offset.y())},
                grid_size,
                height,
                grid_size,
                Fade(cellColors[best_i], 0.55f));
        }
    }
}