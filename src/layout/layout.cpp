#include "layout.h"

namespace layout
{
    SoftRVDModel::SoftRVDModel(
        const torch::Tensor &grid_xy_,
        const torch::Tensor &terrain_h_,
        const torch::Tensor &site_xy_,
        float k_,
        float tau_)
        : grid_xy(grid_xy_),
          terrain_h(terrain_h_),
          site_xy(site_xy_),
          k(k_),
          tau(tau_)
    {
        G = grid_xy.size(0);
        N = site_xy.size(0);

        auto device = grid_xy.device();

        h_cell = register_parameter(
            "h_cell", torch::zeros({N}, torch::requires_grad()));

        weights = torch::zeros({G, N}, device);

        auto gx = grid_xy.index({torch::indexing::Slice(), 0}); // [G]
        auto gy = grid_xy.index({torch::indexing::Slice(), 1}); // [G]

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

                // -------- soft topology score --------
                auto dx = px - qx;
                auto dy = py - qy;

                auto scores = torch::stack({
                    dx - torch::abs(dy),  // left
                    -dx - torch::abs(dy), // right
                    dy - torch::abs(dx),  // front
                    -dy - torch::abs(dx)  // back
                });                       // [4]

                auto alpha = torch::softmax(tau * scores, 0); // [4]

                // -------- half-space mid --------
                auto midx = 0.5f * (px + qx);
                auto midy = 0.5f * (py + qy);

                // -------- weighted half-spaces --------
                logw += alpha[0] * torch::nn::functional::logsigmoid(k * (gx - midx));
                logw += alpha[1] * torch::nn::functional::logsigmoid(k * (midx - gx));
                logw += alpha[2] * torch::nn::functional::logsigmoid(k * (gy - midy));
                logw += alpha[3] * torch::nn::functional::logsigmoid(k * (midy - gy));
            }

            auto w = torch::exp(logw);
            weights.index_put_({torch::indexing::Slice(), i}, w);
        }

        // normalize per grid
        weights = weights / (weights.sum(1, true) + 1e-6);
        auto w_cpu = weights.detach().to(torch::kCPU);

        for (int g = 0; g < grid_xy.size(0); ++g)
        {
            std::cout << "Grid " << g << " weights: ";
            for (int i = 0; i < site_xy.size(0); ++i)
            {
                std::cout << w_cpu[g][i].item<float>() << " ";
            }
            std::cout << std::endl;
        }
    }

    torch::Tensor SoftRVDModel::forward()
    {

        // ===== target height =====
        auto h_target = torch::matmul(weights, h_cell); // [G]

        // ===== earthwork =====
        auto diff = h_target - terrain_h;
        auto E_local = torch::sum(diff * diff);

        auto V = torch::sum(diff);
        auto E_balance = V * V;

        return E_local + 10.0f * E_balance;
    }

    void SoftRVDModel::drawGrids(float z, float size, Eigen::Vector2f offset) const
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
            DrawCube(Vector3{x + offset.x(), z, -(y + offset.y())},size, size, size,c);
            // DrawCubeWires(Vector3{x + offset.x(), z, -(y + offset.y())},size, size, size,RL_BLACK);
        }
    }

}