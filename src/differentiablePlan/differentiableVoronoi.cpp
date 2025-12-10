#include "differentiableVoronoi.h"
#include <torch/torch.h>

torch::Tensor diffVoronoi::Voronoi2Layer::forward(
    torch::autograd::AutogradContext *ctx,
    const torch::Tensor &site2xy,
    const std::vector<float> &vtxl2xy,
    const std::vector<std::array<int64_t, 4>> &vtxv2info)
{
    // 保存反向传播需要的数据
    ctx->saved_data["vtxl2xy"] = vtxl2xy;
    ctx->saved_data["vtxv2info"] = vtxv2info;
    ctx->save_for_backward({site2xy});

    // 获取输入数据
    auto site2xy_contiguous = site2xy.contiguous();
    float *site_ptr = site2xy_contiguous.data_ptr<float>();
    int64_t num_sites = site2xy.size(0);
    std::vector<float> site_data(site_ptr, site_ptr + num_sites * 2);

    // 计算Voronoi顶点
    int64_t num_vtxv = vtxv2info.size();
    torch::Tensor vtxv2xy = torch::empty({num_vtxv, 2}, site2xy.options());
    float *vtxv_ptr = vtxv2xy.data_ptr<float>();

    for (int64_t i_vtxv = 0; i_vtxv < num_vtxv; ++i_vtxv)
    {
        Vec2 cc = util::compute_voronoi_vertex(
            vtxv2info[i_vtxv],
            vtxl2xy,
            site_data);
        vtxv_ptr[i_vtxv * 2] = cc.x;
        vtxv_ptr[i_vtxv * 2 + 1] = cc.y;
    }
    return vtxv2xy;
}

std::vector<torch::Tensor> diffVoronoi::Voronoi2Layer::backward(
    torch::autograd::AutogradContext *ctx,
    std::vector<torch::Tensor> grad_outputs)
{
    // 恢复保存的数据
    auto saved = ctx->get_saved_variables();
    torch::Tensor site2xy = saved[0];

    auto vtxl2xy = ctx->saved_data["vtxl2xy"].to<std::vector<float>>();
    auto vtxv2info = ctx->saved_data["vtxv2info"].to<std::vector<std::array<int64_t, 4>>>();

    torch::Tensor dw_vtxv2xy = grad_outputs[0];

    // 获取输入数据
    auto site2xy_contiguous = site2xy.contiguous();
    float *site_ptr = site2xy_contiguous.data_ptr<float>();
    int64_t num_sites = site2xy.size(0);
    std::vector<float> site_data(site_ptr, site_ptr + num_sites * 2);

    // 获取梯度数据
    auto dw_vtxv2xy_contiguous = dw_vtxv2xy.contiguous();
    float *dw_vtxv_ptr = dw_vtxv2xy_contiguous.data_ptr<float>();
    int64_t num_vtxv = dw_vtxv2xy.size(0);

    // 初始化梯度
    torch::Tensor dw_site2xy = torch::zeros_like(site2xy);
    float *dw_site_ptr = dw_site2xy.data_ptr<float>();

    // 计算每个Voronoi顶点的梯度贡献
    for (int64_t i_vtxv = 0; i_vtxv < num_vtxv; ++i_vtxv)
    {
        const auto &info = vtxv2info[i_vtxv];
        Vec2 dv(dw_vtxv_ptr[i_vtxv * 2], dw_vtxv_ptr[i_vtxv * 2 + 1]);

        if (info[1] == -1)
        { // 边界顶点
            // 无需更新站点梯度
        }
        else if (info[3] == -1)
        { // 边与中垂线交点
            int64_t num_vtxl = vtxl2xy.size() / 2;
            int64_t i1_loop = info[0];
            int64_t i2_loop = (i1_loop + 1) % num_vtxl;

            Vec2 l1 = to_vec2(vtxl2xy, i1_loop);
            Vec2 l2 = to_vec2(vtxl2xy, i2_loop);
            Vec2 ldir = l2 - l1;

            int64_t i0_site = info[1];
            int64_t i1_site = info[2];

            Vec2 s0 = to_vec2(site_data, i0_site);
            Vec2 s1 = to_vec2(site_data, i1_site);

            auto [_, drds0, drds1] = util::dw_intersection_against_bisector(l1, ldir, s0, s1);

            // 计算梯度并累加
            Vec2 ds0(drds0.x * dv.x + drds0.y * dv.y,
                     drds0.x * dv.x + drds0.y * dv.y);
            dw_site_ptr[i0_site * 2] += ds0.x;
            dw_site_ptr[i0_site * 2 + 1] += ds0.y;

            Vec2 ds1(drds1.x * dv.x + drds1.y * dv.y,
                     drds1.x * dv.x + drds1.y * dv.y);
            dw_site_ptr[i1_site * 2] += ds1.x;
            dw_site_ptr[i1_site * 2 + 1] += ds1.y;
        }
        else
        { // 三个Voronoi区域交点
            std::array<int64_t, 3> idx_site = {info[1], info[2], info[3]};
            Vec2 s0 = to_vec2(site_data, idx_site[0]);
            Vec2 s1 = to_vec2(site_data, idx_site[1]);
            Vec2 s2 = to_vec2(site_data, idx_site[2]);

            auto [_, dvds] = util::circumcenter(s0, s1, s2);

            for (int i_node = 0; i_node < 3; ++i_node)
            {
                Vec2 ds(dvds[i_node].x * dv.x + dvds[i_node].y * dv.y,
                        dvds[i_node].x * dv.x + dvds[i_node].y * dv.y);
                int64_t is0 = idx_site[i_node];
                dw_site_ptr[is0 * 2] += ds.x;
                dw_site_ptr[is0 * 2 + 1] += ds.y;
            }
        }
    }

    return {dw_site2xy, torch::Tensor(), torch::Tensor()};
}

// 包装函数
torch::Tensor voronoi2_forward(
    const torch::Tensor &site2xy,
    const std::vector<float> &vtxl2xy,
    const std::vector<std::array<int64_t, 4>> &vtxv2info)
{
    return diffVoronoi::Voronoi2Layer::apply(site2xy, vtxl2xy, vtxv2info);
}
