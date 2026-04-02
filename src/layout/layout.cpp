#include "layout.h"

namespace layout
{
    SoftRVDModel::SoftRVDModel(
        const torch::Tensor &grid_xy_,
        const torch::Tensor &terrain_h_,
        const torch::Tensor &site_xy_,
        float far_,
        const std::vector<int> &courtyard_ids_,
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

        // -------- trainable --------
        floor_logits = register_parameter(
            "floor_logits",
            0.01f * torch::randn({N, 5}, device));

        delta_logits = register_parameter(
            "delta_logits",
            0.01f * torch::randn({N, 5}, device));

        computeWeights();
        register_buffer("weights", weights);
        // -------- base height (最低地基) --------
        auto A = weights.sum(0);
        auto h_site_mean = (weights.transpose(0, 1).matmul(terrain_h)) / (A + 1e-6);

        base_height = h_site_mean.min().detach();
        lloyd_optimizer = std::make_unique<torch::optim::Adam>(
            this->parameters(),
            torch::optim::AdamOptions(0.05));
    }

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

    torch::Tensor SoftRVDModel::forward()
    {
        auto device = grid_xy.device();

        // ===============================
        // 1. site area
        // ===============================
        auto A = weights.sum(0); // [N]

        // ===============================
        // 2. site mean terrain height
        // ===============================
        auto h_site_mean =
            (weights.transpose(0, 1).matmul(terrain_h)) / (A + 1e-6); // [N]

        // ---- global base height (detach!) ----
        auto h_base = h_site_mean.min().detach(); // scalar

        // ===============================
        // 3. floors (soft integer)
        // ===============================
        auto floor_vals = torch::arange(0, 5, device);
        auto p_floor = torch::softmax(floor_logits, 1);
        auto floors = (p_floor * floor_vals).sum(1);

        for (int id : courtyard_ids)
            floors[id] = 0.0f;

        // ===============================
        // 4. discrete dh (relative to base)
        // ===============================
        auto delta_vals = torch::tensor({0.f, 2.f, 4.f, 6.f, 8.f}, device);

        auto p_delta = torch::softmax(delta_logits, 1);
        auto dh = (p_delta * delta_vals).sum(1); // [N]

        // ===============================
        // 5. final site plane height
        // ===============================
        auto H_site = h_base + dh; // [N]

        // ===============================
        // 6. terrain delta (cut & fill)
        // ===============================
        // Δh_g = sum_i w_gi * (H_i - h_g)
        auto delta_terrain =
            weights.matmul(H_site) - terrain_h; // [G]

        auto L_terrain = delta_terrain.pow(2).mean();

        // ===============================
        // 7. FAR loss
        // ===============================
        auto total_floors = (A * floors).sum();
        auto target_floors = G * far;
        auto L_far = (total_floors - target_floors).pow(2);

        // ===============================
        // 8. entropy (anneal)
        // ===============================
        auto L_entropy =
            -(p_floor * torch::log(p_floor + 1e-6)).sum() - (p_delta * torch::log(p_delta + 1e-6)).sum();

        return lambda_far * L_far +
               lambda_terrain * L_terrain +
               lambda_entropy * L_entropy;
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
            DrawCube(Vector3{x + offset.x(), 0, -(y + offset.y())}, size, 0, size, c);
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

    void SoftRVDModel::stepOptimize(int &curIter, const int maxIter)
    {
    }

    void SoftRVDModel::optimize(SoftRVDShowData &showData, int iters, float lr, int verbose_every)
    {
        torch::optim::Adam optimizer(
            this->parameters(),
            torch::optim::AdamOptions(lr));

        for (int iter = 0; iter < iters; ++iter)
        {
            optimizer.zero_grad();

            float t = float(iter) / float(iters - 1);
            float s = t * t * (3.f - 2.f * t);

            lambda_entropy = 0.05f * (1.0f - s);

            auto loss = forward();
            loss.backward();
            optimizer.step();

            // =========================
            // UPDATE SHOW DATA
            // =========================
            {
                auto device = grid_xy.device();

                auto floor_vals = torch::arange(0, 5, device);
                auto p_floor = torch::softmax(floor_logits, 1);
                auto floors_now = (p_floor * floor_vals).sum(1);

                auto delta_vals =
                    torch::tensor({0.f, 2.f, 4.f, 6.f, 8.f}, device);

                auto p_delta = torch::softmax(delta_logits, 1);
                auto dh_now = (p_delta * delta_vals).sum(1);
                dh_now = dh_now * torch::relu(floors_now - 1.0f);

                showData.grid_xy = grid_xy;
                showData.weights = weights;
                showData.floors = floors_now;
                showData.dh = dh_now;
                showData.base_h = base_height;
                showData.G = G;
                showData.N = N;
            }

            if (iter % verbose_every == 0 || iter == iters - 1)
            {
                std::cout
                    << "[Iter " << iter
                    << "] loss=" << loss.item<float>()
                    << std::endl;
            }
        }
    }

    void SoftRVDShowData::draw(float floor_height, float grid_size, Eigen::Vector2f offset) const
    {
        if (!grid_xy.defined())
            return;

        auto w_cpu = weights.detach().to(torch::kCPU);
        auto f_cpu = floors.detach().to(torch::kCPU);
        auto dh_cpu = dh.detach().to(torch::kCPU);
        auto base_cpu = base_h.detach().to(torch::kCPU);

        float base_val = base_cpu.item<float>();

        // ---- site colors ----
        std::vector<Color> cellColors(N);
        for (int i = 0; i < N; ++i)
        {
            float hue = float(i) / float(N);
            cellColors[i] = renderUtil::ColorFromHue(hue);
        }

        // ---- iterate grids ----
        for (int g = 0; g < G; ++g)
        {
            float gx = grid_xy[g][0].item<float>();
            float gy = grid_xy[g][1].item<float>();

            // ---- find dominant site ----
            int best_i = 0;
            float max_w = -1.f;

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
            int floors_int = int(std::round(floors_soft));

            if (floors_int <= 0)
                continue;

            float dh_val = dh_cpu[best_i].item<float>();

            float height = floors_int * floor_height;
            float z_base = base_val + dh_val;

            // 画一个整体高度的 box
            DrawCube(
                Vector3{
                    gx + offset.x(),
                    z_base + height * 0.5f,
                    -(gy + offset.y())},
                grid_size,
                height,
                grid_size,
                cellColors[best_i]);
        }
    }

}