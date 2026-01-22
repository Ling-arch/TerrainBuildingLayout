#include "optimizer.h"

namespace optimizer
{
    PlanProblem define_problem(const int floor, const vector<Vector2> &boundary, const vector<Scalar> &area_ratio, const vector<pair<size_t, size_t>> &room_connections, const vector<Vector2> &fix_sites, const vector<size_t> &fix_rooms)
    {
        PlanProblem plan_prob;
        plan_prob.floor = floor;
        vector<Scalar> vtxl2xy = M2::flat_vec2(boundary);
        // resample polyloop and normalize
        vtxl2xy = polyloop::resample<float, 2>(vtxl2xy, 100);
        plan_prob.vtxl2xy_norm = polyloop::normalize(vtxl2xy, {0.5, 0.5}, 1.f, plan_prob.tf);

        // set room area target
        plan_prob.total_area_trg = M2::polygon_area(plan_prob.vtxl2xy_norm); // normalized total area

        // each room area target
        vector<Scalar> room2area_trg;
        room2area_trg.reserve(area_ratio.size());
        float sum_ratio = 0.f;
        // normalize ratio
        for (float v : area_ratio)
            sum_ratio += v;

        for (float v : area_ratio)
            room2area_trg.push_back(v / sum_ratio * plan_prob.total_area_trg);
        plan_prob.num_room = static_cast<int64_t>(room2area_trg.size());
        plan_prob.room2area_trg = room2area_trg;
        std::cout << " num_room is " << plan_prob.num_room << std::endl;

        // set fix sites and then gennerate possion sites
        vector<Vector2> fix_sites_norm = polyloop::map_pt_normalized(fix_sites, {0.5, 0.5}, plan_prob.tf);
        M2::PoissonResult poisson_result = M2::gen_poisson_sites_in_poly_with_seeds(M2::to_vec2_array(plan_prob.vtxl2xy_norm), fix_sites_norm, Scalar(0.1), 50, (unsigned)time(nullptr));
        plan_prob.sites_norm = poisson_result.samples;
        size_t num_site = plan_prob.sites_norm.size();
        std::cout << "num_site is " << num_site << std::endl;
        vector<float> site2xy2flag(num_site * 2, 0.f);

        if (fix_sites_norm.size() > 0)
        {
            for (size_t idx : poisson_result.seed_indices)
            {
                site2xy2flag[2 * idx + 0] = 1.f;
                site2xy2flag[2 * idx + 1] = 1.f;
            }
        }
        plan_prob.site2room = loss::site2room(
            num_site,
            room2area_trg,
            poisson_result.seed_indices,
            fix_rooms);

        plan_prob.site2xy2flag = site2xy2flag;

        //--------------------------------------tensor define---------------------------------
        plan_prob.options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        plan_prob.site2xy = torch::zeros({static_cast<int64_t>(num_site), 2}, plan_prob.options);

        auto acc = plan_prob.site2xy.accessor<float, 2>();
        for (int i = 0; i < num_site; ++i)
        {
            acc[i][0] = plan_prob.sites_norm[i].x();
            acc[i][1] = plan_prob.sites_norm[i].y();
        }
        plan_prob.site2xy.set_requires_grad(true);
        // site2xy tensor
        plan_prob.site2xy_ini = plan_prob.site2xy.detach().clone();
        plan_prob.optimizer = std::make_unique<torch::optim::AdamW>(
            std::vector<torch::Tensor>{plan_prob.site2xy},
            torch::optim::AdamWOptions(0.05));

        // site2xy2flag
        plan_prob.site2xy2flag_t = torch::from_blob(site2xy2flag.data(), {static_cast<int64_t>(num_site), 2}, plan_prob.options).clone();
        // roomarea tensor
        plan_prob.room2area_trg_t = torch::from_blob(room2area_trg.data(), {static_cast<int64_t>(area_ratio.size()), 1}, plan_prob.options).clone();
        torch::optim::AdamWOptions adamw_opts(0.05);

        return plan_prob;
    }

    void optimize_draw_bystep(PlanProblem &plan_prob, size_t &cur_iter, const size_t max_iter, OptimizeDrawData &voronoi_show)
    {

        int64_t num_site = static_cast<int64_t>(plan_prob.sites_norm.size());
        vector<float> &vtxl2xy_norm = plan_prob.vtxl2xy_norm;
        polyloop::NormalizeTransform2D<Scalar> &tf = plan_prob.tf;
        int64_t num_room = plan_prob.num_room;

        // room area target
        vector<float> &room2area_trg = plan_prob.room2area_trg;

        // site_id to room_id
        vector<size_t> &site2room = plan_prob.site2room;

        // room typology
        vector<pair<size_t, size_t>> &room_connections = plan_prob.room_connections;

        // fixed flags
        vector<float> &site2xy2flag = plan_prob.site2xy2flag;

        // sites
        vector<Vector2> &sites_norm = plan_prob.sites_norm;

        torch::Tensor &site2xy = plan_prob.site2xy;

        // std::cout << "copy data end" << std::endl;
        if (cur_iter >= max_iter)
            return;

        if (cur_iter == 150)
        {
            for (auto &g : plan_prob.optimizer->param_groups())
                static_cast<torch::optim::AdamWOptions &>(g.options()).lr(0.005);
        }

        // ---- Voronoi ----
        auto [vtxv2xy, voronoi_info, site2cell] = diffVoronoi::voronoi(vtxl2xy_norm, site2xy, [&](size_t i_site)
                                                                       { return site2room[i_site] != INVALID; });

        // std::cout << "can calculate " << std::endl;
        vector<size_t> edge2vtxv_wall = loss::edge2vtvx_wall(voronoi_info, site2room);
        // std::cout << "can wall " << std::endl;
        torch::Tensor room2area = loss::room2area(site2room, num_room, voronoi_info.site2idx, voronoi_info.idx2vtxv, vtxv2xy);
        // std::cout << "can area " << std::endl;
        torch::Tensor loss_each_area = (room2area - plan_prob.room2area_trg_t).pow(2).sum();
        // std::cout << "can each area " << std::endl;
        torch::Tensor loss_total_area = (room2area.sum() - torch::tensor(plan_prob.total_area_trg, plan_prob.options)).abs();
        // std::cout << "can total area " << std::endl;
        diffVoronoi::Vtx2XYZToEdgeVectorLayer edge_layer(edge2vtxv_wall);
        torch::Tensor edge2xy = edge_layer.forward(vtxv2xy);
        // std::cout << "can edge2xy " << std::endl;
        torch::Tensor loss_walllen = edge2xy.abs().sum();
        // std::cout << "can wallen " << std::endl;
        torch::Tensor loss_topo = loss::unidirectional(
            site2xy,
            site2room,
            num_room,
            voronoi_info,
            room_connections);
        // std::cout << "can topo " << std::endl;
        torch::Tensor loss_fix = ((site2xy - plan_prob.site2xy_ini) * plan_prob.site2xy2flag_t).pow(2).sum();

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

        // 每25次迭代打印损失信息
        if (cur_iter % 25 == 0)
        {
            std::cout << "Iter " << cur_iter << " - Loss: " << loss.item<float>()
                      << " (areas: " << loss_each_area.item<float>()
                      << ", total_areas: " << loss_total_area.item<float>()
                      << ", topo: " << loss_topo.item<float>()
                      << ", fix: " << loss_fix.item<float>()
                      << ", wall: " << loss_walllen.item<float>() << ")" << std::endl;
        }

        plan_prob.optimizer->zero_grad();
        loss.backward();
        plan_prob.optimizer->step();
        cur_iter++;
        // std::cout << "can BACKWARD " << std::endl;
        // ---------- sites ----------
        const float *site_ptr = site2xy.data_ptr<float>();

        voronoi_show.sites_world.resize(num_site);
        for (int64_t i = 0; i < num_site; ++i)
        {
            Vector2 p{site_ptr[i * 2 + 0], site_ptr[i * 2 + 1]};
            auto denorm = polyloop::denormalize({p}, tf);
            voronoi_show.sites_world[i] =
                polyloop::convert_points_to_3d(
                    denorm,
                    float(plan_prob.floor * 3))[0];
        }

        // ---------- cells ----------
        voronoi_show.cellPolys.clear();
        voronoi_show.cellPolys.reserve(site2cell.size());

        for (const auto &cell : site2cell)
        {
            vector<Vector2> denorm = polyloop::denormalize(cell.vtx2xy, tf);

            voronoi_show.cellPolys.emplace_back(polyloop::convert_points_to_3d(denorm, float(plan_prob.floor * 3)));
        }

        // ---------- walls ----------
        voronoi_show.wall_edge_list.clear();
        voronoi_show.wall_edge_list.reserve(edge2vtxv_wall.size() / 2);

        const float *vtx_ptr = vtxv2xy.data_ptr<float>();

        for (size_t i = 0; i + 1 < edge2vtxv_wall.size(); i += 2)
        {
            size_t i0 = edge2vtxv_wall[i];
            size_t i1 = edge2vtxv_wall[i + 1];

            Vector2 p0{vtx_ptr[i0 * 2 + 0], vtx_ptr[i0 * 2 + 1]};
            Vector2 p1{vtx_ptr[i1 * 2 + 0], vtx_ptr[i1 * 2 + 1]};

            vector<Vector2> denorm = polyloop::denormalize({p0, p1}, tf);

            voronoi_show.wall_edge_list.emplace_back(polyloop::convert_points_to_3d(denorm, float(plan_prob.floor * 3)));
        }
    }

    PlanProblem define_field_problem(const int floor, field::TensorField2D<float> field, const vector<Vector2> &boundary, const vector<float> &area_ratio, const vector<pair<size_t, size_t>> &room_connections, const vector<Vector2> &fix_sites, const vector<size_t> &fix_rooms)
    {
        PlanProblem plan_prob;
        plan_prob.floor = floor;
        vector<Scalar> vtxl2xy = M2::flat_vec2(boundary);
        plan_prob.field = field;

        // resample polyloop and normalize
        vtxl2xy = polyloop::resample<float, 2>(vtxl2xy, 100);
        plan_prob.vtxl2xy_norm = polyloop::normalize(vtxl2xy, {0.5, 0.5}, 1.f, plan_prob.tf);

        // set room area target
        plan_prob.total_area_trg = M2::polygon_area(plan_prob.vtxl2xy_norm); // normalized total area

        // each room area target
        vector<Scalar> room2area_trg;
        room2area_trg.reserve(area_ratio.size());
        float sum_ratio = 0.f;
        // normalize ratio
        for (float v : area_ratio)
            sum_ratio += v;

        for (float v : area_ratio)
            room2area_trg.push_back(v / sum_ratio * plan_prob.total_area_trg);
        plan_prob.num_room = static_cast<int64_t>(room2area_trg.size());
        plan_prob.room2area_trg = room2area_trg;
        std::cout << " num_room is " << plan_prob.num_room << std::endl;

        // set fix sites and then gennerate possion sites
        vector<Vector2> fix_sites_norm = polyloop::map_pt_normalized(fix_sites, {0.5, 0.5}, plan_prob.tf);
        M2::PoissonResult poisson_result = M2::gen_poisson_sites_in_poly_with_seeds(M2::to_vec2_array(plan_prob.vtxl2xy_norm), fix_sites_norm, Scalar(0.08), 50, (unsigned)time(nullptr));
        plan_prob.sites_norm = poisson_result.samples;
        size_t num_site = plan_prob.sites_norm.size();
        std::cout << "num_site is " << num_site << std::endl;
        vector<float> site2xy2flag(num_site * 2, 0.f);

        if (fix_sites_norm.size() > 0)
        {
            for (size_t idx : poisson_result.seed_indices)
            {
                site2xy2flag[2 * idx + 0] = 1.f;
                site2xy2flag[2 * idx + 1] = 1.f;
            }
        }
        plan_prob.site2room = loss::site2room(
            num_site,
            room2area_trg,
            poisson_result.seed_indices,
            fix_rooms);

        plan_prob.site2xy2flag = site2xy2flag;

        //--------------------------------------tensor define---------------------------------
        plan_prob.options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        plan_prob.site2xy = torch::zeros({static_cast<int64_t>(num_site), 2}, plan_prob.options);

        auto acc = plan_prob.site2xy.accessor<float, 2>();
        for (int i = 0; i < num_site; ++i)
        {
            acc[i][0] = plan_prob.sites_norm[i].x();
            acc[i][1] = plan_prob.sites_norm[i].y();
        }
        plan_prob.site2xy.set_requires_grad(true);
        // site2xy tensor
        plan_prob.site2xy_ini = plan_prob.site2xy.detach().clone();
        plan_prob.optimizer = std::make_unique<torch::optim::AdamW>(
            std::vector<torch::Tensor>{plan_prob.site2xy},
            torch::optim::AdamWOptions(0.05));

        // site2xy2flag
        plan_prob.site2xy2flag_t = torch::from_blob(site2xy2flag.data(), {static_cast<int64_t>(num_site), 2}, plan_prob.options).clone();
        // roomarea tensor
        plan_prob.room2area_trg_t = torch::from_blob(room2area_trg.data(), {static_cast<int64_t>(area_ratio.size()), 1}, plan_prob.options).clone();
        torch::optim::AdamWOptions adamw_opts(0.05);

        return plan_prob;
    }

    void optimize_field_problem_and_draw_bystep(PlanProblem &plan_prob, size_t &cur_iter, const size_t max_iter, OptimizeDrawData &voronoi_show)
    {
        int64_t num_site = static_cast<int64_t>(plan_prob.sites_norm.size());
        vector<float> &vtxl2xy_norm = plan_prob.vtxl2xy_norm;
        polyloop::NormalizeTransform2D<Scalar> &tf = plan_prob.tf;
        int64_t num_room = plan_prob.num_room;

        // room area target
        vector<float> &room2area_trg = plan_prob.room2area_trg;

        // site_id to room_id
        vector<size_t> &site2room = plan_prob.site2room;

        // room typology
        vector<pair<size_t, size_t>> &room_connections = plan_prob.room_connections;

        // fixed flags
        vector<float> &site2xy2flag = plan_prob.site2xy2flag;

        // sites
        vector<Vector2> &sites_norm = plan_prob.sites_norm;

        torch::Tensor &site2xy = plan_prob.site2xy;

        // std::cout << "copy data end" << std::endl;
        if (cur_iter >= max_iter)
            return;

        if (cur_iter == 150)
        {
            for (auto &g : plan_prob.optimizer->param_groups())
                static_cast<torch::optim::AdamWOptions &>(g.options()).lr(0.005);
        }

        // ---- Voronoi ----
        auto [vtxv2xy, voronoi_info, site2cell] = diffVoronoi::voronoi(vtxl2xy_norm, site2xy, [&](size_t i_site)
                                                                       { return site2room[i_site] != INVALID; });

        // std::cout << "can calculate " << std::endl;
        vector<size_t> edge2vtxv_wall = loss::edge2vtvx_wall(voronoi_info, site2room);
        // std::cout << "can wall " << std::endl;
        torch::Tensor room2area = loss::room2area(site2room, num_room, voronoi_info.site2idx, voronoi_info.idx2vtxv, vtxv2xy);
        // std::cout << "can area " << std::endl;
        torch::Tensor loss_each_area = (room2area - plan_prob.room2area_trg_t).pow(2).sum();
        // std::cout << "can each area " << std::endl;
        torch::Tensor loss_total_area = (room2area.sum() - torch::tensor(plan_prob.total_area_trg, plan_prob.options)).abs();
        // std::cout << "can total area " << std::endl;
        // diffVoronoi::EdgeTensorAlignLayer edge_layer(edge2vtxv_wall);
        diffVoronoi::Vtx2XYZToEdgeVectorLayer edge_layer(edge2vtxv_wall);
        torch::Tensor edge2xy = edge_layer.forward(vtxv2xy);
        // std::cout << "can edge2xy " << std::endl;
        torch::Tensor loss_walllen = edge2xy.abs().sum();
        torch::Tensor tensor_dir = loss::build_edge_tensor_dir(edge2vtxv_wall,plan_prob.field,vtxv2xy,tf);
        diffVoronoi::EdgeTensorAlignLayer edge_field_layer(edge2vtxv_wall);
        torch::Tensor loss_field = edge_field_layer.forward(vtxv2xy, tensor_dir);
        
        // std::cout << "can edge2xy " << std::endl;

        // std::cout << "can wallen " << std::endl;
        torch::Tensor loss_topo = loss::unidirectional(
            site2xy,
            site2room,
            num_room,
            voronoi_info,
            room_connections);
        // std::cout << "can topo " << std::endl;
        torch::Tensor loss_fix = ((site2xy - plan_prob.site2xy_ini) * plan_prob.site2xy2flag_t).pow(2).sum();

        torch::Tensor loss_lloyd = loss::loss_lloyd(
            voronoi_info.site2idx,
            voronoi_info.idx2vtxv,
            site2xy,
            vtxv2xy);

        loss_each_area = loss_each_area * 1.0;
        loss_total_area = loss_total_area * 10.0;
        loss_field = loss_field * 0.1;
        loss_walllen = loss_walllen * 0.05;
        loss_topo = loss_topo * 1.0;
        loss_fix = loss_fix * 100.0;
        loss_lloyd = loss_lloyd * 0.1;
        torch::Tensor loss = loss_each_area + loss_total_area + loss_field + loss_walllen + loss_topo + loss_fix + loss_lloyd;

        // 每25次迭代打印损失信息
        if (cur_iter % 25 == 0)
        {
            std::cout << "Iter " << cur_iter << " - Loss: " << loss.item<float>()
                      << " (areas: " << loss_each_area.item<float>()
                      << ", total_areas: " << loss_total_area.item<float>()
                      << ", topo: " << loss_topo.item<float>()
                      << ", fix: " << loss_fix.item<float>()
                      << ", wall: " << loss_walllen.item<float>()
                      << ", field: " << loss_field.item<float>() << ")" << std::endl;
        }

        plan_prob.optimizer->zero_grad();
        loss.backward();
        plan_prob.optimizer->step();
        cur_iter++;
        // std::cout << "can BACKWARD " << std::endl;
        // ---------- sites ----------
        const float *site_ptr = site2xy.data_ptr<float>();

        voronoi_show.sites_world.resize(num_site);
        for (int64_t i = 0; i < num_site; ++i)
        {
            Vector2 p{site_ptr[i * 2 + 0], site_ptr[i * 2 + 1]};
            auto denorm = polyloop::denormalize({p}, tf);
            voronoi_show.sites_world[i] =
                polyloop::convert_points_to_3d(
                    denorm,
                    float(plan_prob.floor * 3))[0];
        }

        // ---------- cells ----------
        voronoi_show.cellPolys.clear();
        voronoi_show.cellPolys.reserve(site2cell.size());

        for (const auto &cell : site2cell)
        {
            vector<Vector2> denorm = polyloop::denormalize(cell.vtx2xy, tf);

            voronoi_show.cellPolys.emplace_back(polyloop::convert_points_to_3d(denorm, float(plan_prob.floor * 3)));
        }

        // ---------- walls ----------
        voronoi_show.wall_edge_list.clear();
        voronoi_show.wall_edge_list.reserve(edge2vtxv_wall.size() / 2);

        const float *vtx_ptr = vtxv2xy.data_ptr<float>();

        for (size_t i = 0; i + 1 < edge2vtxv_wall.size(); i += 2)
        {
            size_t i0 = edge2vtxv_wall[i];
            size_t i1 = edge2vtxv_wall[i + 1];

            Vector2 p0{vtx_ptr[i0 * 2 + 0], vtx_ptr[i0 * 2 + 1]};
            Vector2 p1{vtx_ptr[i1 * 2 + 0], vtx_ptr[i1 * 2 + 1]};

            vector<Vector2> denorm = polyloop::denormalize({p0, p1}, tf);

            voronoi_show.wall_edge_list.emplace_back(polyloop::convert_points_to_3d(denorm, float(plan_prob.floor * 3)));
        }
    }
}