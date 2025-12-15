#include "render.h"

#include <iostream>
#include <cmath>

using std::vector;

namespace render
{
    Renderer3D::Renderer3D(int width, int height, float fovy, CameraProjection proj, const char *name)
        : width_(width), height_(height), fovy_(fovy), projection_(proj), name_(name),
          yaw_(0.8f), pitch_(0.4f), distance_(14.f)
    {
        camera_.position = {0.f, 10.f, 10.f};
        camera_.target = {0.f, 0.f, 0.f};
        camera_.up = {0.f, 1.f, 0.f};
        camera_.fovy = fovy_;
        camera_.projection = projection_;

        InitWindow(width_, height_, name);
        SetTargetFPS(120);
    }

    Renderer3D::~Renderer3D()
    {
        CloseWindow();
    }

    void Renderer3D::begin3D()
    {
        if (!in3D)
        {
            BeginMode3D(camera_);
            in3D = true;
        }
    }

    void Renderer3D::end3D()
    {
        if (in3D)
        {
            EndMode3D();
            in3D = false;
        }
    }

    std::shared_ptr<RenderPoly> Renderer3D::createPolyMesh()
    {
        auto poly = std::make_shared<RenderPoly>();
        polys.push_back(poly);
        return poly;
    }

    void Renderer3D::updateCamera()
    {
        Vector2 md = GetMouseDelta();

        // 左键旋转
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        {
            yaw_ += md.x * RENDER_ROTATE_SPEED;
            pitch_ += md.y * RENDER_ROTATE_SPEED;

            // 限制 pitch 防止抖动
            float limit = PI / 2 - 0.01f;
            if (pitch_ > limit)
                pitch_ = limit;
            if (pitch_ < -limit)
                pitch_ = -limit;
        }

        // 滚轮：缩放（沿视线方向）
        float wheel = GetMouseWheelMove();
        if (wheel != 0)
        {
            distance_ -= wheel * RENDER_ZOOM_SPEED;
            if (distance_ < 1.f)
                distance_ = 1.f;
        }

        // 右键或者滚轮按下：平移
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))
        {
            Vector3 forward = {
                cosf(pitch_) * sinf(yaw_),
                sinf(pitch_),
                cosf(pitch_) * cosf(yaw_)};
            Vector3 up = {0, 1, 0};
            Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, up));
            up = Vector3Normalize(Vector3CrossProduct(right, forward));

            float dx = md.x * RENDER_MOVE_SPEED;
            float dy = md.y * RENDER_MOVE_SPEED;

            Vector3 pan = Vector3Add(
                Vector3Scale(right, dx),
                Vector3Scale(up, dy));
            camera_.target = Vector3Add(camera_.target, pan);
        }

        Vector3 offset = {
            distance_ * cosf(pitch_) * sinf(yaw_),
            distance_ * sinf(pitch_),
            distance_ * cosf(pitch_) * cosf(yaw_)};
        camera_.position = Vector3Add(camera_.target, offset);
        camera_.up = {0, 1, 0};
    }

    void Renderer3D::fill_polygon2(RenderPoly &poly, const std::vector<Vec2> &pts, Scalar z, Color color, float alpha, bool doublesided)
    {
        vector<Vector3> pts_vector3 = Renderer3D::vec2_to_Vector3(pts, float(z));
        fill_polygon3(poly, pts_vector3, color, alpha, doublesided);
    }

    void Renderer3D::fill_polygon3(RenderPoly &poly, const std::vector<Vector3> &pts, Color color, float alpha, bool doublesided)
    {
        poly.triangles.clear();
        poly.dirty = false;
        if (pts.size() < 3)
            return;
        // 1. 计算平面
        Vector3 origin, normal;
        if (!compute_plane(pts, origin, normal))
            return;

        // 2. 构造平面正交基
        Vector3 u, v;
        make_plane_basis(normal, u, v);

        vector<Vec2> polygon = project_to_2d(pts, origin, u, v);
        auto polygon_earcut = M2::convert_to_earcut(polygon);
        // earcut 返回的是索引序列，3个一组是三角形
        std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(polygon_earcut);

        if (indices.empty())
            return;

        Color final_color = Fade(color, alpha);

        for (size_t i = 0; i + 2 < indices.size(); i += 3)
        {
            Triangle3 tri;
            tri.points = {
                pts[indices[i]],
                pts[indices[i + 1]],
                pts[indices[i + 2]]};
            tri.color = final_color;
            tri.doubleSided = doublesided;
            poly.triangles.emplace_back(tri);
        }
    }

    void Renderer3D::stroke_polygon2(const std::vector<Vec2> &pts, Scalar z, float thickness, Color color, float alpha)
    {
        vector<Vector3> pts_vector3 = Renderer3D::vec2_to_Vector3(pts, float(z));
        stroke_polygon3(pts_vector3, thickness, color, alpha);
    }

    void Renderer3D::stroke_polygon3(const vector<Vector3> &pts, float thickness, Color color, float alpha)
    {
        for (size_t i = 0; i < pts.size(); ++i)
            DrawCylinderEx(pts[i], pts[(i + 1) % pts.size()], thickness, thickness, 1, Fade(color, alpha));
    }

    void draw_points(const std::vector<Vec2> &pts, Scalar z, Scalar radius, Color color, float alpha)
    {
    }

    void Renderer3D::draw_faces()
    {

        for (auto &poly : polys)
        {
            for (auto &tri : poly->triangles)
            {
                DrawTriangle3D(
                    tri.points[0],
                    tri.points[1],
                    tri.points[2],
                    tri.color);
                if (tri.doubleSided)
                {
                    DrawTriangle3D(
                        tri.points[0],
                        tri.points[2],
                        tri.points[1],
                        tri.color);
                }
            }
        }
    }

    void Renderer3D::runMainLoop(const FrameCallbacks &callBack)
    {
        while (!WindowShouldClose())
        {
            updateCamera();
            if (callBack.onUpdate)
                callBack.onUpdate();
            BeginDrawing();
            ClearBackground(RAYWHITE);
            BeginMode3D(camera_);
            draw_faces();
            if (callBack.onDraw3D)
                callBack.onDraw3D();
            EndMode3D();
            if (callBack.onDraw2D)
                callBack.onDraw2D();
            EndDrawing();
        }
    }

    void Renderer3D::clear()
    {
        render_triangles.clear();
    }

    Vector3 Renderer3D::vec2_to_Vector3(const Vec2 &v2, float z)
    {
        return {v2.x(), z, -v2.y()};
    }

    vector<Vector3> Renderer3D::vec2_to_Vector3(const vector<Vec2> &arr2, float z)
    {
        vector<Vector3> arr3;
        arr3.reserve(arr2.size());
        for (const auto &p2 : arr2)
            arr3.push_back(vec2_to_Vector3(p2, z));
        return arr3;
    }

    bool Renderer3D::compute_plane(
        const std::vector<Vector3> &pts,
        Vector3 &origin,
        Vector3 &normal,
        float eps)
    {
        if (pts.size() < 3)
            return false;

        origin = pts[0];

        // 找不共线的三点
        Vector3 n = {0, 0, 0};
        for (size_t i = 1; i + 1 < pts.size(); ++i)
        {
            Vector3 a = Vector3Subtract(pts[i], origin);
            Vector3 b = Vector3Subtract(pts[i + 1], origin);
            n = Vector3CrossProduct(a, b);
            if (Vector3Length(n) > eps)
                break;
        }

        if (Vector3Length(n) <= eps)
            return false; // 全共线

        normal = Vector3Normalize(n);

        // 检查所有点到平面的距离
        for (auto &p : pts)
        {
            float d = Vector3DotProduct(
                Vector3Subtract(p, origin), normal);
            if (fabsf(d) > eps)
                return false; // 不共面
        }

        return true;
    }

    void Renderer3D::make_plane_basis(
        const Vector3 &n,
        Vector3 &u,
        Vector3 &v)
    {
        Vector3 tmp = (fabsf(n.x) < 0.9f)
                          ? Vector3{1, 0, 0}
                          : Vector3{0, 1, 0};

        u = Vector3Normalize(Vector3CrossProduct(tmp, n));
        v = Vector3CrossProduct(n, u);
    }

    std::vector<Vec2> Renderer3D::project_to_2d(
        const std::vector<Vector3> &pts,
        const Vector3 &origin,
        const Vector3 &u,
        const Vector3 &v)
    {
        std::vector<Vec2> out;
        out.reserve(pts.size());

        for (auto &p : pts)
        {
            Vector3 d = Vector3Subtract(p, origin);
            Vec2 p;
            p << Vector3DotProduct(d, u),
                Vector3DotProduct(d, v);
            out.push_back(p);
        }
        return out;
    }
}