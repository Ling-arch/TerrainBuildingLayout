#include "render.h"
#include "polygonMesh.h"
#include "polyloop.h"
#include "optimizer.h"
#include "diffVoronoi.h"
#include "voronoi2.h"
#include "loss.h"

using render::Renderer3D;
using Scalar = render::Scalar;
using M2 = util::Math2<Scalar>;
using Vec2 = typename M2::Vector2;
using Vec3 = typename M2::Vector3;
using Mat2 = typename M2::Matrix2;
using polyloop::Polyloop2, polyloop::Polyloop3;
using std::cout, std::endl, std::vector, std::array;

inline constexpr size_t INVALID = static_cast<size_t>(std::numeric_limits<int64_t>::max());

static Color room_color_from_id(size_t room_id, size_t num_room)
{
    if (room_id == INVALID)
        return BLACK;
    float t = float(room_id) / float(std::max<size_t>(1, num_room));
    float h = t * 360.0f; // hue
    float s = 0.6f;
    float v = 0.9f;

    float c = v * s;
    float x = c * (1 - std::fabsf(std::fmod(h / 60.0f, 2) - 1));
    float m = v - c;

    float r = 0, g = 0, b = 0;
    if (h < 60)
    {
        r = c;
        g = x;
    }
    else if (h < 120)
    {
        r = x;
        g = c;
    }
    else if (h < 180)
    {
        g = c;
        b = x;
    }
    else if (h < 240)
    {
        g = x;
        b = c;
    }
    else if (h < 300)
    {
        r = x;
        b = c;
    }
    else
    {
        r = c;
        b = x;
    }
    return Color{(unsigned char)((r + m) * 255), (unsigned char)((g + m) * 255), (unsigned char)((b + m) * 255), 255};
}

struct CellDrawData{
    Polyloop2 polyloop;
    size_t site_id;
};

int main()
{
    render::FrameCallbacks cb;
    Renderer3D render(1920, 1080, 45.0f, CAMERA_PERSPECTIVE, "FloorPlanTest");
    vector<Vec2> boundary = {
        {-5.12f, -2.62f},
        {27.68f, -2.62f},
        {27.68f, 9.42f},
        {21.8f, 9.42f},
        {21.8f, 18.85f},
        {19.78f, 18.85f},
        {19.78f, 25.44f},
        {10.08f, 25.44f},
        {10.08f, 23.48f},
        {0.f, 23.48f},
        {0.f, 6.59f},
        {-5.12f, 6.59f}};

    // boundary flat
    vector<float> vtxl2xy_origin = M2::flat_vec2(boundary);
    vector<float> vtxl2xy = polyloop::resample<float, 2>(vtxl2xy_origin, 100);

    polyloop::NormalizeTransform2D<float> tf;
    /* std::vector<float> */ 
    vtxl2xy = polyloop::normalize<float>(vtxl2xy, std::array<float, 2>{0.5f, 0.5f}, 1.f, tf);
    vector<float> denormalize_vtxl2xy = polyloop::denormalize(vtxl2xy,tf);
     const int64_t num_room = 4;
    const vector<float> area_ratio = {0.4f, 0.2f, 0.2f, 0.2f};
    vector<Vec2> sites_norm = M2::gen_poisson_sites_in_poly(M2::to_vec2_array(vtxl2xy), Scalar(0.1), 50, (unsigned)time(nullptr));
    vector<Vec2> denormalized_sites = M2::to_vec2_array(polyloop::denormalize(M2::flat_vec2(sites_norm), tf));
    int64_t num_site = static_cast<int64_t>(sites_norm.size());
    vector<float> room2area_trg;
    float total_area_trg = M2::polygon_area(M2::to_vec2_array(vtxl2xy));

    float sum_ratio = 0.f;
    for (float v : area_ratio)
        sum_ratio += v;

    // compute target area per room
    room2area_trg.reserve(area_ratio.size());
    for (float v : area_ratio)
        room2area_trg.push_back(v / sum_ratio * total_area_trg);

    vector<size_t> site2room = loss::site2room(num_site, room2area_trg); // demo
  
    vector<float> site2xy2flag(num_site * 2, 0.f);
    vector<std::pair<size_t, size_t>> room_connections = {{0, 1}, {0, 2}, {0, 3}};
    // std::cout << "room_connections has set " << std::endl;
    //  // 给最后一个房间，加上固定 site
    //  sites.push_back( {0.1f, 0.15f} );
    //  site2xy2flag.insert(site2xy2flag.end(),2, 1.f);
    //  site2room.push_back(room2area_trg.size() - 1);

    // sites.push_back({2.f, 0.15f});
    // site2xy2flag.insert(site2xy2flag.end(), 2, 1.f);
    // site2room.push_back(room2area_trg.size() - 1);
    // num_site = static_cast<int64_t>(sites.size());

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor site2xy = torch::zeros({num_site, 2}, options);
    {
        auto acc = site2xy.accessor<float, 2>();
        for (int i = 0; i < num_site; ++i)
        {
            acc[i][0] = sites_norm[i].x();
            acc[i][1] = sites_norm[i].y();
        }
    }
    site2xy.set_requires_grad(true);

    // site2xy tensor
    torch::Tensor site2xy_ini = site2xy.detach().clone();
    // site2xy2flag
    torch::Tensor site2xy2flag_t = torch::from_blob(site2xy2flag.data(), {num_site, 2}, options).clone();
    // roomarea tensor
    torch::Tensor room2area_trg_t = torch::from_blob(room2area_trg.data(), {num_room, 1}, options).clone();
    torch::optim::AdamWOptions adamw_opts(0.05);
    torch::optim::AdamW optimizer({site2xy}, adamw_opts);
    vector<Polyloop2> cellPolys;
    vector<CellDrawData> cellPolyDatas;
    vector<Vector3> site_world_pos;

    int iter = 0;
    const int max_iter = 250;
    bool optimizing = true;

    diffVoronoi::VoronoiInfo current_voronoi_info;
    torch::Tensor current_vtxv2xy;
    vector<size_t> edge2vtxv_wall;
    vector<vector<Vec2>> wall_edge_list;

    //----------------------------相当于draw部分------------------------
    render.runMainLoop(render::FrameCallbacks{
        [&]() { // 按键更新，重新绘图等事件，poly修改过需要重新fill
            if (!optimizing || iter >= max_iter)
                return;

            if (iter == 150)
            {
                for (auto &g : optimizer.param_groups())
                    static_cast<torch::optim::AdamWOptions &>(g.options()).lr(0.005);
            }

            // ---- Voronoi ----
            auto [vtxv2xy, voronoi_info] = diffVoronoi::voronoi(vtxl2xy, site2xy, [&](size_t i_site)
                                                                { return site2room[i_site] != INVALID; });
            // std::cout << "voronoi_info has caculated " << std::endl;
            //  wall edges
            auto site2cell = voronoi2::voronoi_cells(vtxl2xy, M2::flat_vec2(sites_norm), [&](size_t i_site){ return site2room[i_site] != INVALID; });


            edge2vtxv_wall = loss::edge2vtvx_wall(voronoi_info, site2room);
            // 保存结果用于绘图
            current_voronoi_info = voronoi_info;
            current_vtxv2xy = vtxv2xy;
            // ---- 直接从diffVoronoi结果构造绘图多边形（避免重复计算）----
            vector<float> current_vtxv2xy_norm = diffVoronoi::flat_tensor_to_float(vtxv2xy);
            auto& site2idx = voronoi_info.site2idx;
            auto& idx2vtxv = voronoi_info.idx2vtxv;
            cellPolyDatas.clear();
            for (size_t i_site = 0; i_site + 1 < voronoi_info.site2idx.size(); ++i_site)
            {
                if (site2room[i_site] == INVALID)
                    continue;

                // 获取该站点的顶点环
                size_t src = site2idx[i_site];
                size_t end = site2idx[i_site + 1];

                vector<float> flat_cell_vertices;

                for (size_t idx = src; idx < end; ++idx)
                {
                    size_t i0 = idx2vtxv[idx];
                    // 从vtxv2xy获取顶点坐标
                    float x = current_vtxv2xy_norm[i0 * 2 + 0];
                    float y = current_vtxv2xy_norm[i0 * 2 + 1];
                    flat_cell_vertices.push_back(x);
                    flat_cell_vertices.push_back(y);
                }
                vector<float> denormalized_cell_vertices = polyloop::denormalize(flat_cell_vertices, tf);
                if (!denormalized_cell_vertices.empty() && denormalized_cell_vertices.size() % 2 == 0)
                {
                    Polyloop2 cell(M2::to_vec2_array(denormalized_cell_vertices));
                    cellPolyDatas.push_back({cell, i_site});
                }
            }

            cellPolys.clear();
            cellPolys.reserve(site2cell.size());
            for (auto &cell : site2cell)
            {
                vector<float> denormalized_cell_vertices = polyloop::denormalize(M2::flat_vec2(cell.vtx2xy), tf);
                cellPolys.emplace_back(M2::to_vec2_array(denormalized_cell_vertices));
            }
            

            wall_edge_list.clear();
            for (size_t i = 0; i < edge2vtxv_wall.size(); i += 2)
            {
                size_t vtx_i0 = edge2vtxv_wall[i];
                size_t vtx_i1 = edge2vtxv_wall[i + 1];
                float x0 = vtxv2xy.accessor<float, 2>()[vtx_i0][0];
                float y0 = vtxv2xy.accessor<float, 2>()[vtx_i0][1];
                float x1 = vtxv2xy.accessor<float, 2>()[vtx_i1][0];
                float y1 = vtxv2xy.accessor<float, 2>()[vtx_i1][1];
                vector<Vec2> edge;
                vector<float> edge_flat = {
                    x0, y0,
                    x1, y1};
                edge_flat = polyloop::denormalize(edge_flat, tf);

                edge.push_back({edge_flat[0], edge_flat[1]});
                edge.push_back({edge_flat[2], edge_flat[3]});
                wall_edge_list.push_back(edge);
            }

 
            auto room2area = loss::room2area(site2room, num_room, voronoi_info.site2idx, voronoi_info.idx2vtxv, vtxv2xy);

            torch::Tensor loss_each_area = (room2area - room2area_trg_t).pow(2).sum();

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

            // 每25次迭代打印损失信息
            if (iter % 25 == 0)
            {
                std::cout << "Iter " << iter << " - Loss: " << loss.item<float>()
                          << " (areas: " << loss_each_area.item<float>()
                          << ", wall: " << loss_walllen.item<float>() << ")" << std::endl;
            }
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            // std::cout << "backward succeed at :" << iter << std::endl;
            //  ---- Tensor → CPU sites ----
            torch::Tensor site2xy_cpu = site2xy.detach().cpu().contiguous();
            auto acc = site2xy_cpu.accessor<float, 2>();
            for (int i = 0; i < num_site; ++i)
            {
                sites_norm[i].x() = acc[i][0];
                sites_norm[i].y() = acc[i][1];
            }

            // ---- recompute cells (draw) ----
            vector<float> denormalized_sites2xy = polyloop::denormalize(M2::flat_vec2(sites_norm), tf);
            denormalized_sites = M2::to_vec2_array(denormalized_sites2xy);

            site_world_pos.clear();
            for (auto &s : denormalized_sites)
                site_world_pos.push_back({s.x(), 0.5f, -s.y()});

            ++iter;
        },
        [&]() { // 3维空间绘图内容部分
            for (size_t i = 0; i < cellPolyDatas.size(); i++)
            {
                Polyloop2 cell = cellPolyDatas[i].polyloop;
                // size_t room_id = site2room[i];
                render.stroke_light_polygon2(cell, BLACK, 0.f);
                render.fill_polygon2(cell, room_color_from_id(site2room[cellPolyDatas[i].site_id], num_room), 0.0f, 0.5f);
                render.draw_points(denormalized_sites, RED);
            }
            render.stroke_bold_polygon2(Polyloop2(M2::to_vec2_array(denormalize_vtxl2xy)), BLACK,0.F,0.07f);
            render.draw_points(M2::to_vec2_array(denormalize_vtxl2xy),RED,1.f,0.2f,0.f);
           
            for (size_t i = 0; i < wall_edge_list.size(); i ++)
            {
                render.draw_bold_polyline2(wall_edge_list[i], BLACK, 0.f,0.06f);
            }
        },
        [&]() { // 二维屏幕空间绘图
            DrawText(TextFormat("iter = %d", iter), 10, 10, 20, BLACK);
            render.draw_index_fonts(site_world_pos, 21, BLUE);
        }});

    return 0;
}