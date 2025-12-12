#include "util.h"
#include <Eigen/Dense>
#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include "voronoi2.h"
#include <random>
#include <vector>
using Eigen::Matrix2d;
using Eigen::Vector2d,Eigen::Vector3f;
using std::cout, std::endl,std::vector;

#define MOVE_SPEED 0.01f
#define ROTATE_SPEED 0.003f
#define ZOOM_SPEED 1.0f



inline Vector3 vec2_to_vec3(const Vector2d &v2, float z)
{
    return {static_cast<float>(v2.x()), 0.0f, -static_cast<float>(v2.y())};
}


static vector<Vector3> vector2d_arr_to_vec3(const vector<Vector2d> &arr2, float z)
{
    vector<Vector3> arr3;
    arr3.reserve(arr2.size());
    for (const auto &p2 : arr2)
    {
        arr3.push_back(vec2_to_vec3(p2, z));
    }
    return arr3;
}


// ---------- generate some random sites inside bounding box ----------
static vector<Vector2d> gen_random_sites(size_t n, double minx, double maxx, double miny, double maxy, unsigned seed = 0)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dx(minx, maxx);
    std::uniform_real_distribution<double> dy(miny, maxy);
    vector<Vector2d> sites;
    sites.reserve(n);
    for (size_t i = 0; i < n; ++i)
        sites.emplace_back(dx(rng), dy(rng));
    return sites;
}


// ---------- draw a filled polygon on XZ plane (y=0) using triangle fan ----------
static void draw_polygon3d_fill(const vector<Vector3> &pts, Color fillColor,float alpha)
{
    if (pts.size() < 3)
        return;
    // pick centroid
    Vector3 cen = {0, 0, 0};
    for (auto &p : pts)
        cen = Vector3Add(cen, p);
    cen = Vector3Scale(cen, 1.0f / (float)pts.size());
    for (size_t i = 0; i + 1 < pts.size(); ++i)
    {
        DrawTriangle3D(pts[0], pts[i + 1], pts[(i + 2) % pts.size()], Fade(fillColor, alpha));
    }
}

static void draw_polygon3d_stroke(const vector<Vector3> &pts,float strokeWidth,Color strokeColor,float alpha)
{
    for (size_t i = 0; i < pts.size(); ++i)
    {
        DrawCylinderEx(pts[i], pts[(i + 1) % pts.size()], strokeWidth, strokeWidth, 1, Fade(strokeColor, alpha));
    }
}

void draw3D_window(int width, int height, float fovy, CameraProjection projection)
{
    // ---------- voronoi data ----------
    // define bounding polygon (loop) in 2D (clockwise or ccw)
    vector<Vector2d> boundary = {
        {-20.0, -12.0},
        {20.0, -12.0},
        {20.0, 12.0},
        {-20.0, 12.0}};
    // random sites inside box
    size_t num_sites = 20;
    vector<Vector2d> sites = gen_random_sites(num_sites, -18.0, 18.0, -10.0, 10.0, 12345);

    // helper to compute cells
    auto compute_voronoi = [&]() -> vector<voronoi2::Cell>
    {
        vector<double> vtxl2xy_flat = util::flat_vec2(boundary);
        vector<double> site2xy_flat = util::flat_vec2(sites);
        auto alive = [&](size_t idx) -> bool
        { return true; };
        auto cells = voronoi2::voronoi_cells(vtxl2xy_flat, site2xy_flat, alive);
        return cells;
    };

    vector<voronoi2::Cell> site2cell = compute_voronoi();

    // flag to recompute when R pressed
    bool recompute = false;


    InitWindow(width, height, "3D Stable Orbit Camera");
    SetTargetFPS(120);

    Camera3D camera = {0};
    camera.position = {0.0f, 10.0f, 10.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = fovy;
    camera.projection = projection;

    //------------------------------------
    // 轨道相机参数（核心）
    //------------------------------------
    float yaw = 0.8f;       // yaw 初始值
    float pitch = 0.4f;    // pitch 初始值
    float distance = 14.0f; // camera 到 target 的距离

    while (!WindowShouldClose())
    {
        Vector2 md = GetMouseDelta();
        // recompute voronoi when pressing R (also randomize sites for demo)
        if (IsKeyPressed(KEY_R))
        {
            sites = gen_random_sites(num_sites, -18.0, 18.0, -10.0, 10.0, (unsigned)time(nullptr));
            site2cell = compute_voronoi();
        }


        //------------------------------------
        // 左键：旋转（Orbit）
        //------------------------------------
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        {
            yaw += md.x * ROTATE_SPEED;
            pitch += md.y * ROTATE_SPEED;

            // 限制 pitch 防止抖动
            float limit = PI / 2 - 0.01f;
            if (pitch > limit)
                pitch = limit;
            if (pitch < -limit)
                pitch = -limit;
        }

        //------------------------------------
        // 滚轮：缩放（沿视线方向）
        //------------------------------------
        float wheel = GetMouseWheelMove();
        if (wheel != 0)
        {
            distance -= wheel * ZOOM_SPEED;
            if (distance < 1.0f)
                distance = 1.0f;
        }

        //------------------------------------
        // 右键：平移（Pan）
        //------------------------------------
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
        {
            // forward 向量（球坐标转换）
            Vector3 forward = {
                cosf(pitch) * sinf(yaw),
                sinf(pitch),
                cosf(pitch) * cosf(yaw)};
            Vector3 up = {0, 1, 0};
            Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, up));
            up = Vector3Normalize(Vector3CrossProduct(right, forward));

            float dx = md.x * MOVE_SPEED;
            float dy = md.y * MOVE_SPEED;

            Vector3 pan = Vector3Add(
                Vector3Scale(right, dx),
                Vector3Scale(up, dy));

            camera.target = Vector3Add(camera.target, pan);
        }

        //------------------------------------
        // 用 yaw / pitch / distance 更新 camera.position
        //------------------------------------
        Vector3 offset = {
            distance * cosf(pitch) * sinf(yaw),
            distance * sinf(pitch),
            distance * cosf(pitch) * cosf(yaw)};

        camera.position = Vector3Add(camera.target, offset);
        camera.up = {0, 1, 0}; // 固定 up（非常重要）

        //------------------------------------
        // 绘制
        //------------------------------------
        BeginDrawing();
        ClearBackground(RAYWHITE);
        

        BeginMode3D(camera);

        DrawGrid(10, 1.0f);

        // XYZ 轴
        DrawLine3D({0, 0, 0}, {10000, 0, 0}, RED);
        DrawLine3D({0, 0, 0}, {0, 10000, 0}, GREEN);
        DrawLine3D({0, 0, 0}, {0, 0, 10000}, BLUE);

    
        vector<Vector3> boundary_vec3 = vector2d_arr_to_vec3(boundary, 0.0f);
        //draw_polygon3d_fill(boundary_vec3, YELLOW, 0.1f);
        draw_polygon3d_stroke(boundary_vec3, 0.1f, BLACK, 0.75f);

        // draw each voronoi cell
        Color cell_edge_color = BLACK;
        Color cell_fill_color = BLUE;
        Color site_color = RED;

        for (size_t si = 0; si < site2cell.size(); ++si)
        {
            const voronoi2::Cell &c = site2cell[si];
            if (c.vtx2xy.empty())
                continue;
            // convert to Vector3 list
            vector<Vector3> poly3 = vector2d_arr_to_vec3(c.vtx2xy, 0.0f);
            draw_polygon3d_fill(poly3, cell_fill_color, 0.1f);
            draw_polygon3d_stroke(poly3, 0.05f, cell_edge_color, 0.35f);
        }

        // draw sites
        for (size_t i = 0; i < sites.size(); ++i)
        {
            Vector3 p3 = vec2_to_vec3(sites[i], 0.0f);
            DrawSphere(p3, 0.05f, RED);
        }

        EndMode3D();

        DrawText("Camera (Left: Rotate, Right: Pan, Wheel: Zoom)", 10, 10, 20, DARKGRAY);

        EndDrawing();
    }

    CloseWindow();
}

int main()
{
    Vector2d p0(-1, -1);
    Vector2d p1(1, 1);
    Vector2d ls(-1, 1);
    Vector2d ld(1, -1);

    Vector2d r;
    Matrix2d drda, drdb;
    std::tie(r, drda, drdb) = util::dw_intersection(p0, p1, ls, ld);

    cout << "Intersection point: " << r.transpose() << endl;
    cout << "Derivative wrt a: \n"
         << drda << endl;
    cout << "Derivative wrt b: \n"
         << drdb << endl;

    Vector3 movement = {0.0f, 0.0f, 0.0f};
    Vector3 rotation = {0.0f, 0.0f, 0.0f};
    float zoom = 0.0f;

    

    draw3D_window(1920, 1080, 45.0f, CAMERA_PERSPECTIVE);

    return 0;
}