#include "render.h"
#include "renderUtil.h"
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
using namespace optimizer;


inline constexpr size_t INVALID = static_cast<size_t>(std::numeric_limits<int64_t>::max());


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
    const vector<float> area_ratio = {0.4f, 0.3f, 0.2f, 0.1f, 0.1f, 0.1f};
    const vector<std::pair<size_t,size_t>> room_connections = {{0,1},{0,2},{1,3},{0,4},{4,5}};

    PlanProblem plan_prob = define_problem(0, boundary, area_ratio, room_connections, {}, {});
    std::vector<Color> room2colors = renderUtil::room2colors(area_ratio.size());
    OptimizeDrawData draw_data;
    
    // vector<vector<Vec3>> wall_edge_list = draw_data.wall_edge_list;
    size_t cur_iter = 0;
    auto start = std::chrono::high_resolution_clock::now();
    bool is_optimizing = false;

    auto optimize_step_safe = [&]()
    {
        try
        {
            optimize_draw_bystep(plan_prob, cur_iter, 250, draw_data);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Optimization failed: " << e.what() << std::endl;
            return false;
        }
    };
    //----------------------------相当于draw部分------------------------
    rlImGuiSetup(true);
    render.runMainLoop(render::FrameCallbacks{
        [&]() { // 按键更新，重新绘图等事件，poly修改过需要重新fill
            if (IsKeyPressed(KEY_R))
            {
                plan_prob = define_problem(0, boundary, area_ratio, room_connections, {}, {});
                is_optimizing = true;
                cur_iter = 0;
                
            }

            if (is_optimizing)
            {
                bool ok = optimize_step_safe();
                if (!ok)
                {
                    plan_prob =  define_problem(0, boundary, area_ratio, room_connections, {}, {});
                    cur_iter = 0;
                    is_optimizing = true; // 或 false
                }
            }
        },
        [&]() { // 3维空间绘图内容部分
            render::stroke_bold_polygon2(Polyloop2(boundary), RL_BLACK, 0.f, 0.03f);
            for (size_t i = 0; i < draw_data.cellPolys.size(); i++)
            {
                render::stroke_light_polygon3(draw_data.cellPolys[i], RL_BLACK);
                render::fill_polygon3(draw_data.cellPolys[i], room2colors[plan_prob.site2room[i]],0.8f);
            }
            render::draw_points(draw_data.sites_world, RL_RED);

            for (size_t i = 0; i < draw_data.wall_edge_list.size(); i++)
            {
                render::draw_bold_polyline3(draw_data.wall_edge_list[i], RL_BLACK, 0.06f);
            }
        },
        [&]() { // 二维屏幕空间绘图
            rlImGuiBegin();
            DrawText(TextFormat("iter = %d", cur_iter), 10, 10, 20, RL_BLACK);
            // render.draw_index_fonts(site_world_pos, 21, BLUE);
            rlImGuiEnd();
        }});

    return 0;
}