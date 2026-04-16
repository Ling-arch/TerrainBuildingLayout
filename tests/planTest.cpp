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

        //多线段的凹边界
    vector<Vec2> boundary1 = {
        {0.f, 0.f},
        {18.f, 0.f},
        {18.f, 8.f},
        {12.f, 8.f},
        {12.f, 15.f},
        {4.f, 15.f},
        {4.f, 10.f},
        {0.f, 10.f}};

    // 异形轮廓
    vector<Vec2> boundary2 = {
        {-5.484741f, 9.606076f},
        {-6.741274f, 9.330292f},
        {-7.764197f, 8.556481f},
        {-8.370428f, 7.426352f},
        {-8.340885f, -0.311724f},
        {-7.696708f, -1.420547f},
        {-6.648366f, -2.159147f},
        {-5.382426f, -2.393885f},
        {-4.098986f, -2.527596f},
        {-2.983403f, -3.159053f},
        {-2.233237f, -4.198207f},
        {-1.983632f, -5.463380f},
        {-1.678515f, -6.714776f},
        {-0.880964f, -7.718636f},
        {0.263329f, -8.298101f},
        {8.015259f, -8.393924f},
        {9.271792f, -8.118140f},
        {10.294716f, -7.344330f},
        {10.900947f, -6.214201f},
        {10.871404f, 1.523875f},
        {10.227227f, 2.632699f},
        {9.178885f, 3.371299f},
        {7.912945f, 3.606037f},
        {6.629504f, 3.739747f},
        {5.513922f, 4.371204f},
        {4.763755f, 5.410358f},
        {4.514150f, 6.675531f},
        {4.209033f, 7.926927f},
        {3.411483f, 8.930787f},
        {2.267190f, 9.510252f},
        {0.980853f, 9.606064f},
        {-5.484741f, 9.606076f} // 闭合点，与第一个点相同
    };

    std::vector<Vec2> livingRoomPt_1 = {{11.5f,14.5f}};
    std::vector<Vec2> livingRoomPt_2 = {{-7.877f,6.765f}};

    //客厅，餐厅，厨房，卫生间，卧室，书房,入口
    const vector<float> area_ratio = {0.4f, 0.3f, 0.2f, 0.1f, 0.1f, 0.1f/* ,0.05f */};
    //客厅-厨房，客厅-厨房，餐厅-卫生间，客厅-卧室，卧室-书房
    const vector<std::pair<size_t,size_t>> room_connections = {{0,1},{0,2},{1,3},{0,4},{4,5}/* ,{0,6} */};

    PlanProblem plan_prob = define_problem(0, boundary1, area_ratio, room_connections, livingRoomPt_1, {0});
    std::vector<Color> room2colors = renderUtil::room2colors(area_ratio.size());
    OptimizeDrawData draw_data;
    std::vector<Eigen::Vector2f> deNormBounds = M2::to_vec2_array(plan_prob.vtxl2xy_norm);
    std::vector<Eigen::Vector2f> rects = {{1.f,1.f},{1.f,-1.f},{-1.f,-1.f},{-1.f,1.f}};
    // vector<vector<Vec3>> wall_edge_list = draw_data.wall_edge_list;
    size_t cur_iter = 0;
    auto start = std::chrono::high_resolution_clock::now();
    bool is_optimizing = false;
    // std::vector<Eigen::Vector2f> resampledPoly = M2::to_vec2_array(polyloop::resample<float, 2>(M2::flat_vec2(boundary), 100));
    auto optimize_step_safe = [&]()
    {
        try
        {
            optimize_draw_bystep(plan_prob, cur_iter, 251, draw_data);
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
                plan_prob = define_problem(0, boundary1, area_ratio, room_connections, livingRoomPt_1, {0});
                is_optimizing = true;
                cur_iter = 0;
                
            }

            if (is_optimizing)
            {
                bool ok = optimize_step_safe();
                if (!ok)
                {
                    plan_prob = define_problem(0, boundary1, area_ratio, room_connections, livingRoomPt_1, {0});
                    cur_iter = 0;
                    is_optimizing = true; // 或 false
                }
            }
        },
        [&]() { // 3维空间绘图内容部分
            render::stroke_bold_polygon2(Polyloop2(boundary1), RL_BLACK, 0.f, 0.03f);
            // render::stroke_bold_polygon2(resampledPoly, RL_BLACK, 0.f, 0.03f,1.f,{20.f,0.f});
            // render::stroke_bold_polygon2(deNormBounds, RL_BLACK, 0.f, 0.01f, 1.f, {40.f, 0.f});
            // render::draw_points(resampledPoly,RL_RED,1.0F,0.2F,0.f,{20.f,0.f});
            // render::draw_points(deNormBounds, RL_RED, 1.0F, 0.015F, 0.f, {40.f, 0.f});
        
            for (size_t i = 0; i < draw_data.cellPolys.size(); i++)
            {
                Color c = renderUtil::ColorFromHue((float)plan_prob.site2room[i] /plan_prob.num_room);
                render::stroke_light_polygon3(draw_data.cellPolys[i], RL_BLACK);
                render::fill_polygon3(draw_data.cellPolys[i], c,0.3f);
            }
            render::draw_points(draw_data.sites_world, RL_RED);

            for (size_t i = 0; i < draw_data.wall_edge_list.size(); i++)
            {
                render::draw_bold_polyline3(draw_data.wall_edge_list[i], RL_BLACK, 0.06f);
            }

            
        },
        [&]() { // 二维屏幕空间绘图
            
            rlImGuiBegin();
            render.setCameraUI(true);
            DrawText(TextFormat("iter = %d", cur_iter), 10, 10, 20, RL_BLACK);
            // render.draw_index_fonts(site_world_pos, 21, BLUE);
            rlImGuiEnd();
        }});

    return 0;
}