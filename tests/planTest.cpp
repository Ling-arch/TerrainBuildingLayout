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
using std::cout, std::endl, std::vector;

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
    if (h < 60){
        r = c;
        g = x;
    }
    else if (h < 120){
        r = x;
        g = c;
    }
    else if (h < 180){
        g = c;
        b = x;
    }
    else if (h < 240){
        g = x;
        b = c;
    }
    else if (h < 300){
        r = x;
        b = c;
    }
    else{
        r = c;
        b = x;
    }
    return Color{(unsigned char)((r + m) * 255),(unsigned char)((g + m) * 255),(unsigned char)((b + m) * 255),255};
}

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
    std::vector<float> vtxl2xy_origin = M2::flat_vec2(boundary);
    std::vector<float> vtxl2xy = polyloop::resample<float, 2>(vtxl2xy_origin, 100);

    polyloop::NormalizeTransform2D<float> tf;
    /* std::vector<float> */ vtxl2xy = polyloop::normalize<float>(vtxl2xy, std::array<float, 2>{0.5f, 0.5f}, 1.f, tf);
    const int64_t num_room = 4;
    const std::vector<float> area_ratio = {0.4f, 0.2f, 0.2f, 0.2f};
    vector<Vec2> sites_norm = M2::gen_poisson_sites_in_poly(M2::to_vec2_array(vtxl2xy), Scalar(0.1), 50, (unsigned)time(nullptr));
    vector<Vec2> denormalized_sites = M2::to_vec2_array(polyloop::denormalize(M2::flat_vec2(sites_norm), tf));
    int64_t num_site = static_cast<int64_t>(sites_norm.size());
    std::vector<float> room2area_trg;
    float total_area_trg = M2::polygon_area(M2::to_vec2_array(vtxl2xy));

    // total area of boundary polygon
    // float total_area = M2::polygon_area(boundary); // 或你自己的 area 函数
    std::cout << "total_area = " << total_area_trg << std::endl;

    // sum of ratios
    float sum_ratio = 0.f;
    for (float v : area_ratio)
        sum_ratio += v;

    // compute target area per room
    room2area_trg.reserve(area_ratio.size());
    for (float v : area_ratio)
        room2area_trg.push_back(v / sum_ratio * total_area_trg);

    std::cout << "room2area_trg = " << room2area_trg << std::endl;
    std::vector<size_t> site2room = loss::site2room(num_site, room2area_trg); // demo
    // std::cout << "site2room has caculated " << std::endl;
    // std::cout << "site2room size is : " << site2room.size()<<std::endl;
    // std::cout << "sites size is : " << sites_norm.size() << std::endl;
    // for (int i = 0; i < sites_norm.size(); i++)
    // {
    //     std ::cout << " site :" << i << " room_id is :" << site2room[i] << std::endl;
    //     /* code */
    // }
    
    
    std::vector<float> site2xy2flag(num_site * 2, 0.f);
    std::vector<std::pair<size_t, size_t>> room_connections = {{0, 1}, {0, 2}, {0, 3}};
    //std::cout << "room_connections has set " << std::endl;
    // // 给最后一个房间，加上固定 site
    // sites.push_back( {0.1f, 0.15f} );
    // site2xy2flag.insert(site2xy2flag.end(),2, 1.f);
    // site2room.push_back(room2area_trg.size() - 1);
    
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
    std::vector<Polyloop2> cellPolys;

    std::vector<Vector3> site_world_pos;

    int iter = 0;
    const int max_iter = 250;
    bool optimizing = true;

    //----------------------------相当于draw部分------------------------
    render.runMainLoop(
        render::FrameCallbacks{
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
                //std::cout << "voronoi_info has caculated " << std::endl;
                // wall edges
                std::vector<size_t> edge2vtxv_wall = loss::edge2vtvx_wall(voronoi_info, site2room);
                //std::cout << "edge2vtxv_wall has caculated " << std::endl;
                auto room2area = loss::room2area(site2room, num_room, voronoi_info.site2idx, voronoi_info.idx2vtxv, vtxv2xy);
                //std::cout << "room2area has caculated " << std::endl;
                torch::Tensor loss_each_area = (room2area - room2area_trg_t).pow(2).sum();
                //std::cout << "loss_each_area has caculated " << std::endl;
                torch::Tensor loss_total_area = (room2area.sum() - torch::tensor(total_area_trg, options)).abs();
                //std::cout << "loss_total_area has caculated " << std::endl;
                diffVoronoi::Vtx2XYZToEdgeVectorLayer edge_layer(edge2vtxv_wall);
                torch::Tensor edge2xy = edge_layer.forward(vtxv2xy);
                //std::cout << "edge2xy has caculated " << std::endl;
                torch::Tensor loss_walllen = edge2xy.abs().sum();
                //std::cout << "loss_walllen has caculated " << std::endl;
                torch::Tensor loss_topo = loss::unidirectional(
                    site2xy,
                    site2room,
                    num_room,
                    voronoi_info,
                    room_connections);
                //std::cout << "loss_topo has caculated " << std::endl;

                //torch::Tensor loss_fix = ((site2xy - site2xy_ini) * site2xy2flag_t).pow(2).sum();

                torch::Tensor loss_lloyd = loss::loss_lloyd(
                    voronoi_info.site2idx,
                    voronoi_info.idx2vtxv,
                    site2xy,
                    vtxv2xy);

                loss_each_area = loss_each_area * 1.0;
                loss_total_area = loss_total_area * 10.0;
                loss_walllen = loss_walllen * 0.02;
                loss_topo = loss_topo * 1.0;
                // loss_fix = loss_fix * 100.0;
                loss_lloyd = loss_lloyd * 0.1;
                torch::Tensor loss = loss_each_area + loss_total_area + loss_walllen + loss_topo /* + loss_fix */ +  loss_lloyd;

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
                //std::cout << "backward succeed at :" << iter << std::endl;
                // ---- Tensor → CPU sites ----
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
                auto cells = voronoi2::voronoi_cells(
                    vtxl2xy_origin,
                    denormalized_sites2xy,
                    [&](size_t i_site)
                    { return site2room[i_site] != INVALID; });
                //std::cout << "cells has caculated" << iter << std::endl;
                cellPolys.clear();

                for (auto &c : cells)
                {
                    cellPolys.emplace_back(c.vtx2xy);
                }

                site_world_pos.clear();
                for (auto &s : denormalized_sites)
                    site_world_pos.push_back({s.x(), 0.5f, -s.y()});

                ++iter;
            },
            [&]() { // 3维空间绘图内容部分（前面设置了fill可以不用再绘制）
                for (size_t i = 0; i < cellPolys.size(); i++)
                {
                    Polyloop2 cell = cellPolys[i];
                    // size_t room_id = site2room[i];
                    render.stroke_polygon2(cell, BLACK, 0.f, 0.03f);
                    render.fill_polygon2(cell, room_color_from_id(site2room[i],num_room), 0.0f, 0.5f);
                    render.draw_points(denormalized_sites, RED);
                }

            },
            [&]() { // 二维屏幕空间绘图
                DrawText(TextFormat("iter = %d", iter), 10, 10, 20, BLACK);
                render.draw_index_fonts(site_world_pos, 21, BLUE);
            }});

    return 0;
}