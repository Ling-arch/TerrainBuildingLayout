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

void print_flat_float_array(const std::vector<float> &arr, const std::string &name = "array")
{
    std::cout << name << " (flat float[], 大小: " << arr.size() << "):" << std::endl;
    std::cout << "{";

    if (arr.size() % 2 != 0)
    {
        std::cout << "warn: array size %2 not zero!" << std::endl;
    }

    for (size_t i = 0; i < arr.size(); i += 2)
    {
        if (i + 1 < arr.size())
        {
            std::cout << "{" << arr[i] << "," << arr[i + 1] << "}";
        }
        else
        {
            std::cout << "{" << arr[i] << ",???}"; // 最后一个坐标不完整
        }

        if (i + 2 < arr.size())
        {
            std::cout << ",";
        }
    }
    std::cout << "}" << std::endl;
}

void print_vec2_array(const std::vector<Vec2> &arr, const std::string &name = "array")
{
    std::cout << name << " (vector<Vec2>, size: " << arr.size() << "):" << std::endl;
    std::cout << "{";
    for (size_t i = 0; i < arr.size(); ++i)
    {
        std::cout << "{" << arr[i].x() << "," << arr[i].y() << "}";
        if (i < arr.size() - 1)
        {
            std::cout << ",";
        }
    }
    std::cout << "}" << std::endl;
}

static Color room_color_from_id(size_t room_id, size_t num_room)
{
    if (room_id == INVALID)
        return RL_BLACK;
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
    vector<Polyloop3> cellPolys3d;
    vector<Polyloop2> diffPolys;
    std::vector<Vector3> vtxv_world_pos;
    std::vector<Vector3> vtxv_diff_pos;
    std::vector<Vector3> site_world_pos;
    std::vector<Vec3> site_real_pos;
    std::vector<Vec3> vtxv_real_pos;
};

static void buildVoronoi(
    const vector<Vec2> &boundary,
    const vector<float> &vtxl2xy,
    const vector<float> &room2area_trg,
    VoronoiDrawResult &result_out)
{
    vector<Vec2> sites = M2::gen_poisson_sites_in_poly(boundary, Scalar(3.5), 50, (unsigned)time(nullptr));
    vector<float> site2xy_arr = M2::flat_vec2(sites);
    vector<size_t> site2room = loss::site2room(sites.size(), room2area_trg);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor site2xy = torch::from_blob(site2xy_arr.data(), {static_cast<int64_t>(sites.size()), 2}, options).clone();
    auto [vtxv2xy, voronoi_info,site2cell] = diffVoronoi::voronoi(vtxl2xy, site2xy,
                                                        [&](size_t i_site)
                                                        { return site2room[i_site] != INVALID; });

  

    vector<float> current_vtxv2xy_norm = diffVoronoi::flat_tensor_to_float(vtxv2xy);
    vector<Vec2> diff_vtxv2xy = M2::to_vec2_array(current_vtxv2xy_norm);
    std::vector<Vector3> vtxv_diff_pos;

    vtxv_diff_pos.reserve(sites.size());
    for (auto &vtx : diff_vtxv2xy)
        vtxv_diff_pos.push_back({vtx.x(), 0.5f, -vtx.y()});

    auto &site2idx = voronoi_info.site2idx;
    auto &idx2vtxv = voronoi_info.idx2vtxv;
    vector<Polyloop2> diffPolys;
    for (size_t i_site = 0; i_site + 1 < voronoi_info.site2idx.size(); ++i_site)
    {
        if (site2room[i_site] == INVALID)
            continue;

        // 获取该站点的顶点环
        size_t src = site2idx[i_site];
        size_t end = site2idx[i_site + 1];

        vector<Vec2> diff_cell_vertices;
        std::cout << "\n=====================================" << std::endl;
        std::cout << "[Site " << i_site << "] vertex ring info is:" << std::endl;
        std::cout << "  indices range from src = " << src << ", end = " << end << std::endl;
        std::cout << "  vertex num is :" << (end - src) << std::endl;
        std::cout << "  vertex indices are [";
        for (size_t idx = src; idx < end; ++idx)
        {
            size_t i0 = idx2vtxv[idx];
            // 从vtxv2xy获取顶点坐标
            float x = current_vtxv2xy_norm[i0 * 2 + 0];
            float y = current_vtxv2xy_norm[i0 * 2 + 1];
            Vec2 v = {x, y};
            diff_cell_vertices.push_back(v);

            if (idx > src)
                std::cout << ", ";
            std::cout << i0;
        }
        std::cout << "]" << std::endl;
        diffPolys.emplace_back(diff_cell_vertices);
    }
    vector<Polyloop2> cellPolys;
    vector<Polyloop3> cellPolys3d;
    cellPolys.reserve(site2cell.size());
    cellPolys3d.reserve(site2cell.size());
    
    for (auto &cell : site2cell){
        cellPolys.emplace_back(cell.vtx2xy);
        vector<Vec3> pts3d = polyloop::convert_points_to_3d(cellPolys[cellPolys.size() - 1].points(),0);
        cellPolys3d.emplace_back(pts3d);
    }

    for (int i = 0; i < cellPolys.size(); i++)
    {
        std::cout << "cur 2d cell " << i << " tri is ";
        for (auto &tri : cellPolys[i].triangles())
        {
            std::cout << "{" << tri.at(0) << " , " << tri.at(1) << " , " << tri.at(2) << "} ,";
        }
        std::cout << std::endl;
    }
    for(int i = 0 ; i < cellPolys3d.size(); i++){
        float signed_proj_area = M2::signed_polygon_area(cellPolys3d[i].projected_points());
        std::cout << "cur 3d cell " << i << " project signed area is " << signed_proj_area << std::endl;
        std::cout << "cur 3d cell " << i << " tri is ";
        for (auto &tri : cellPolys3d[i].triangles())
        {
            std::cout << "{" << tri.at(0) << " , " << tri.at(1) << " , " << tri.at(2) << "} ,";
        }
        std::cout << std::endl;
    }
    voronoi2::VoronoiMesh mesh = voronoi2::indexing(site2cell);
    auto &vtxv2xy_arr = mesh.vtxv2xy;
    vector<Vector3> vtxv_world_pos;
    vector<Vector3> site_world_pos;
    std::vector<Vec3> site_real_pos;
    std::vector<Vec3> vtxv_real_pos;
    print_vec2_array(vtxv2xy_arr, "mesh vtxv2xy");
    site_world_pos.reserve(sites.size());
    site_real_pos.reserve(sites.size());
    for (auto &site : sites)
    {
        site_world_pos.push_back({site.x(), 0.5f, -site.y()});
        site_real_pos.push_back({site.x(), 0.f, -site.y()});
    }

    vtxv_world_pos.reserve(vtxv2xy_arr.size());
    vtxv_real_pos.reserve(vtxv2xy_arr.size());
    for (auto &vtx : vtxv2xy_arr)
    {
        vtxv_world_pos.push_back({vtx.x(), 0.5f, -vtx.y()});
        vtxv_real_pos.push_back({vtx.x(), vtx.y(), 0.f});
    }

    result_out.sites = sites;
    result_out.site2room = site2room;
    result_out.cellPolys = cellPolys;
    result_out.diffPolys = diffPolys;
    result_out.vtxv_world_pos = vtxv_world_pos;
    result_out.site_world_pos = site_world_pos;
    result_out.vtxv_real_pos = vtxv_real_pos;
    result_out.site_real_pos = site_real_pos;
    result_out.vtxv_diff_pos = vtxv_diff_pos;
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
    buildVoronoi(boundary, vtxl2xy, room2area_trg, result);
    Vec2 s0 = {0.f, 2.f};
    Vec2 s1 = {2.f, 0.f};
    Vec2 s2 = {0.f, 0.f};

    Vec2 circumcenter = M2::circumcenter(s0,s1,s2);
    std::cout << "test circumcenter is {" << circumcenter.x() <<" , " << circumcenter.y() << "}" <<endl;
    std::vector<Vec2> test_boundary = {{0.f,0.f},{1.f,0.f},{1.f,1.f},{0.f,1.f}};
    std::vector<Vec2> test_sites = M2::gen_poisson_sites_in_poly(test_boundary, Scalar(0.15), 15, (unsigned)time(nullptr));
    diffVoronoi::test_backward_cpp_exact(test_boundary, test_sites);

    //----------------------------相当于draw部分------------------------
    render.runMainLoop(
        render::FrameCallbacks{
            [&]() { // 按键更新，重新绘图等事件，poly修改过需要重新fill
                if (IsKeyPressed(KEY_R))
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    buildVoronoi(boundary, vtxl2xy, room2area_trg, result);
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
                    Polyloop3 cell3d = Polyloop3(polyloop::convert_points_to_3d(cell.points(),0.f));
                    render::stroke_light_polygon2(cell, RL_BLACK, 0.f);
                    // render.fill_polygon2(cell, room_color_from_id(result.site2room[i], num_room), 0.0f, 0.5f);
                    render::fill_polygon3(cell3d, RL_GREEN, 0.5F);
                }

                for (size_t i = 0; i < result.diffPolys.size(); i++)
                {
                    Polyloop2 cell = result.diffPolys[i];
                    // size_t room_id = site2room[i];
                    render::stroke_light_polygon2(cell, RL_GREEN, 0.f);
                    // render.fill_polygon2(cell, room_color_from_id(result.site2room[i], num_room), 0.0f, 0.5f);
                }
                render::draw_points(result.sites, RL_RED);
                render::draw_points(result.vtxv_real_pos, RL_RED, 0.5f, 0.1f);

            },
            [&]() { // 二维屏幕空间绘图
                DrawText("Hello", 10, 10, 20, RL_BLACK);
                render.draw_index_fonts(result.site_world_pos, 16, RL_BLUE);
                render.draw_index_fonts(result.vtxv_world_pos, 20, RL_RED);
                render.draw_index_fonts(result.vtxv_diff_pos, 20, RL_DARKBLUE);
            }});

    return 0;
}