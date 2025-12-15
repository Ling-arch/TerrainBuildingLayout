#include "render.h"

using render::Renderer3D;
using Scalar = render::Scalar;
using M2 = util::Math2<Scalar>;
using Vec2 = typename M2::Vector2;
using Mat2 = typename M2::Matrix2;
using std::cout, std::endl, std::vector;

int main()
{
    render::FrameCallbacks cb;
    Renderer3D render(1920, 1080, 45.0f, CAMERA_PERSPECTIVE, "Render");
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

    auto boundaryPoly = render.createPolyMesh();
    render.fill_polygon2(*boundaryPoly, boundary, Scalar(0), BLUE, 0.5f);

    vector<Vector3> plan = {
        {0.f, 0.f, 0},
        {10.0f, 0.0f, 0},
        {10.0f, 10.0f, 3},
        {0.f, 10.0f, 3}};

    auto planPoly = render.createPolyMesh();
    render.fill_polygon3(*planPoly, plan, GREEN, 0.5f, true);

    //----------------------------相当于draw部分------------------------
    render.runMainLoop(
        render::FrameCallbacks{
            [&]() { // 按键更新，重新绘图等事件，poly修改过需要重新fill
                if (IsKeyPressed(KEY_R))
                    render.fill_polygon2(*boundaryPoly, boundary, 0, RED, 0.5f);
            },
            [&]() { // 3维空间绘图内容部分（前面设置了fill可以不用再绘制）
                DrawGrid(10, 1);

            },
            [&]() { // 二维屏幕空间绘图
                DrawText("Hello", 10, 10, 20, BLACK);
            }});

    return 0;
}