#include "optimizer.h"

namespace optimizer
{
    using M2 = util::Math2<float>;
    torch::Tensor optimize(
        std::vector<float> vtxl2xy,
        std::vector<float> site2xy_init,
        std::vector<size_t> site2room,
        std::vector<float> site2xy2flag,
        std::vector<float> room2area_trg,
        std::vector<std::pair<size_t, size_t>> room_connections)
    {
        const int64_t num_site = static_cast<int64_t>(site2xy_init.size() / 2);

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        // site2xy (requires grad)
        torch::Tensor site2xy = torch::from_blob(site2xy_init.data(), {num_site, 2}, options).clone().set_requires_grad(true);

        // site2xy2flag
        torch::Tensor site2xy2flag_t = torch::from_blob(site2xy2flag.data(), {num_site, 2}, options).clone();

        // initial copy
        torch::Tensor site2xy_ini = site2xy.detach().clone();

        // room2area_trg (num_room, 1)
        const int64_t num_room = room2area_trg.size();
        torch::Tensor room2area_trg_t = torch::from_blob(room2area_trg.data(), {num_room, 1}, options).clone();
        torch::optim::AdamWOptions adamw_opts(0.05);
        torch::optim::AdamW optimizer({site2xy}, adamw_opts);
        for (int iter = 0; iter < 250; ++iter)
        {
            if (iter == 150)
            {
                for (auto &group : optimizer.param_groups())
                {
                    static_cast<torch::optim::AdamWOptions &>(group.options()).lr(0.005);
                }
            }

            // --------------------------------------------------
            // Voronoi
            auto [vtxv2xy, voronoi_info] = diffVoronoi::voronoi(vtxl2xy, site2xy, [&](size_t i_site)
                                                                { return site2room[i_site] != INVALID; });

            // wall edges
            std::vector<size_t> edge2vtxv_wall = loss::edge2vtvx_wall(voronoi_info, site2room);
            auto room2area = loss::room2area(site2room, num_room, voronoi_info.site2idx, voronoi_info.idx2vtxv, vtxv2xy);

            torch::Tensor loss_each_area = (room2area - room2area_trg_t).pow(2).sum();

            float total_area_trg = M2::polygon_area(M2::to_vec2_array(vtxl2xy));

            torch::Tensor loss_total_area = (room2area.sum() - torch::tensor(total_area_trg, options)).abs();
            diffVoronoi::Vtx2XYZToEdgeVectorLayer edge_layer(edge2vtxv_wall);
            torch::Tensor edge2xy = edge_layer.forward(vtxv2xy);

            torch::Tensor loss_walllen = edge2xy.abs().sum();
            torch::Tensor loss_topo = loss::unidirectional(
                site2xy,
                site2room,
                num_room,
                voronoi_info,
                room_connections);

            torch::Tensor loss_fix = ((site2xy - site2xy_ini) * site2xy2flag_t).pow(2).sum();

            torch::Tensor loss_lloyd = loss::loss_lloyd(
                voronoi_info.site2idx,
                voronoi_info.idx2vtxv,
                site2xy,
                vtxv2xy);

            loss_each_area = loss_each_area * 1.0;
            loss_total_area = loss_total_area * 10.0;
            loss_walllen = loss_walllen * 0.02;
            loss_topo = loss_topo * 1.0;
            loss_fix = loss_fix * 100.0;
            loss_lloyd = loss_lloyd * 0.1;
            torch::Tensor loss = loss_each_area + loss_total_area + loss_walllen + loss_topo + loss_fix + loss_lloyd;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        return site2xy;
    }
}