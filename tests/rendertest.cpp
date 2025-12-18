#include "render.h"
#include "polyloop.h"
#include "loss.h"
#include <chrono>
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

struct VoronoiDrawResult
{
    vector<Vec2> sites;
    vector<size_t> site2room;
    vector<voronoi2::Cell> site2cell;
    vector<Polyloop2> cellPolys;
    std::vector<Vector3> vtxv_world_pos;
    std::vector<Vector3> site_world_pos;
};

static void buildVoronoi(
    const vector<Vec2> &boundary,
    const vector<float> &vtxl2xy,
    const vector<float> &room2area_trg,
    VoronoiDrawResult &result_out)
{
    vector<Vec2> sites = M2::gen_poisson_sites_in_poly(boundary, Scalar(2), 30, (unsigned)time(nullptr));
    vector<float> site2xy_arr = M2::flat_vec2(sites);
    vector<size_t> site2room = loss::site2room(sites.size(), room2area_trg);
    vector<voronoi2::Cell> site2cell = voronoi2::voronoi_cells(M2::flat_vec2(boundary), M2::flat_vec2(sites), [&](size_t i_site)
                                                               { return site2room[i_site] != INVALID; });
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor site2xy = torch::from_blob(site2xy_arr.data(), {static_cast<int64_t>(sites.size()), 2}, options).clone();
    auto [vtxv2xy, voronoi_info] = diffVoronoi::voronoi(vtxl2xy, site2xy,
                                                        [&](size_t i_site)
                                                        { return site2room[i_site] != INVALID; });


    for (size_t i_site = 0; i_site + 1 < voronoi_info.site2idx.size(); ++i_site)
    {
        if (site2room[i_site] == INVALID)
            continue; // 跳过无效 site（和 alive 一致）

        size_t beg = voronoi_info.site2idx[i_site];
        size_t end = voronoi_info.site2idx[i_site + 1];

        std::cout << "Site " << i_site << " (room " << site2room[i_site] << ") cell vtxv idx: " << endl;

        for (size_t k = beg; k < end; ++k)
        {
            size_t v0 = voronoi_info.idx2vtxv[k];
            size_t v1 = voronoi_info.idx2vtxv[(k + 1 < end) ? (k + 1) : beg];

            std::cout << "    (" << v0 << " -> " << v1 << ")\n";
        }
        std::cout << "\n";
    }

    vector<size_t> edge2vtvx_wall = loss::edge2vtvx_wall(voronoi_info,site2room);
    cout << "Wall edges: ";
    for (size_t i = 0; i < edge2vtvx_wall.size(); i += 2)
    {
        if (i < edge2vtvx_wall.size() - 1)
        {
            cout << "{" << edge2vtvx_wall[i] << "," << edge2vtvx_wall[i + 1] << "}";
            if (i + 2 < edge2vtvx_wall.size())
            {
                cout << ", ";
            }
        }
    }
    cout << endl;

    
    vector<Polyloop2>cellPolys;
    cellPolys.reserve(site2cell.size());
    for (auto &cell : site2cell)
        cellPolys.emplace_back(cell.vtx2xy);
    voronoi2::VoronoiMesh mesh = voronoi2::indexing(site2cell);
    auto &vtxv2xy_arr = mesh.vtxv2xy;
    vector<Vector3> vtxv_world_pos;
    vector<Vector3> site_world_pos;

    site_world_pos.reserve(sites.size());
    for (auto &site : sites)
        site_world_pos.push_back({site.x(), 0.5f, -site.y()});

    vtxv_world_pos.reserve(vtxv2xy_arr.size());
    for (auto &vtx : vtxv2xy_arr)
        vtxv_world_pos.push_back({vtx.x(), 0.5f, -vtx.y()});
    result_out.sites=sites;
    result_out.site2room = site2room;
    result_out.cellPolys = cellPolys;
    result_out.vtxv_world_pos = vtxv_world_pos;
    result_out.site_world_pos = site_world_pos;
}

int main()
{
    render::FrameCallbacks cb;
    Renderer3D render(1920, 1080, 45.0f, CAMERA_PERSPECTIVE, "VoronoiRender");
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


    std::vector<float> vtxl2xy = M2::flat_vec2(boundary);
    const int64_t num_room = 6;
    const std::vector<float> area_ratio = {0.2f, 0.4f, 0.2f, 0.2f, 0.2f, 0.1f};
    std::vector<float> room2area_trg;
    float total_area_trg = M2::polygon_area(boundary);
    std::cout << "total_area = " << total_area_trg << std::endl;

    float sum_ratio = 0.f;
    for (float v : area_ratio)
        sum_ratio += v;

    // compute target area per room
    room2area_trg.reserve(area_ratio.size());
    for (float v : area_ratio)
        room2area_trg.push_back(v / sum_ratio * total_area_trg);
    VoronoiDrawResult result;
    buildVoronoi(boundary, vtxl2xy,room2area_trg, result);

    //----------------------------相当于draw部分------------------------
    render.runMainLoop(
        render::FrameCallbacks{
            [&]() { // 按键更新，重新绘图等事件，poly修改过需要重新fill
                if (IsKeyPressed(KEY_R))
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    buildVoronoi(boundary, vtxl2xy,room2area_trg, result);
                    auto end = std::chrono::high_resolution_clock::now();

                    // 计算时间差
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

                    std::cout << "Excute time: " << duration.count() << " ms" << std::endl;
                }
            },
            [&]() { // 3维空间绘图内容部分
                for (size_t i = 0; i < result.cellPolys.size(); i++)
                {
                    Polyloop2 cell = result.cellPolys[i];
                    // size_t room_id = site2room[i];
                    render.stroke_polygon2(cell, BLACK, 0.f, 0.03f);
                    render.fill_polygon2(cell, room_color_from_id(result.site2room[i], num_room), 0.0f, 0.5f);
                    render.draw_points(result.sites, RED);
                }

            },
            [&]() { // 二维屏幕空间绘图
                DrawText("Hello", 10, 10, 20, BLACK);
                render.draw_index_fonts(result.site_world_pos, 21, BLUE);
                render.draw_index_fonts(result.vtxv_world_pos, 21, RED);
            }});

    return 0;
}