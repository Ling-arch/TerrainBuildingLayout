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
        camera_.position = {50.f, 50.f, 50.f};
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

    void Renderer3D::fill_polygon2(const Polyloop2 &poly, Color color, float z, float alpha, bool doubleSided)
    {
        if (poly.triangles().size() <= 0)
            return;
        for (auto &tri : poly.triangles())
        {
            Vec2 a = poly.points()[tri.at(0)];
            Vec2 b = poly.points()[tri.at(1)];
            Vec2 c = poly.points()[tri.at(2)];
            DrawTriangle3D(
                {a.x(), z, -a.y()},
                {b.x(), z, -b.y()},
                {c.x(), z, -c.y()},
                Fade(color, alpha));

            // DrawLine3D(tri.points[0], tri.points[1], RED);
            // DrawLine3D(tri.points[1], tri.points[2], BLUE);
            // DrawLine3D(tri.points[2], tri.points[0], GREEN);
            if (doubleSided)
            {
                DrawTriangle3D(
                    {a.x(), z, -a.y()},
                    {c.x(), z, -c.y()},
                    {b.x(), z, -b.y()},
                    Fade(color, alpha));
            }
        }
    }

    void Renderer3D::fill_polygon3(const Polyloop3 &poly, Color color, float alpha, bool doubleSided)
    {
        if (poly.triangles().size() <= 0)
            return;
        for (auto &tri : poly.triangles())
        {
            Vec3 a = poly.points()[tri.at(0)];
            Vec3 b = poly.points()[tri.at(1)];
            Vec3 c = poly.points()[tri.at(2)];
            DrawTriangle3D(
                {a.x(), a.z(), -a.y()},
                {b.x(), b.z(), -b.y()},
                {c.x(), c.z(), -c.y()},
                Fade(color, alpha));

            // DrawLine3D(tri.points[0], tri.points[1], RED);
            // DrawLine3D(tri.points[1], tri.points[2], BLUE);
            // DrawLine3D(tri.points[2], tri.points[0], GREEN);
            if (doubleSided)
            {
                DrawTriangle3D(
                    {a.x(), a.z(), -a.y()},
                    {c.x(), c.z(), -c.y()},
                    {b.x(), b.z(), -b.y()},
                    Fade(color, alpha));
            }
        }
    }

    void Renderer3D::stroke_light_polygon2(const Polyloop2 &poly, Color color, float z, float alpha)
    {
        vector<Vector3> pts_vector3 = vec2_to_Vector3_arr(poly.points(), float(z));
        size_t n = pts_vector3.size();
        for (size_t i = 0; i < n; ++i)
        {
            DrawLine3D(pts_vector3[i], pts_vector3[(i + 1) % n], Fade(color, alpha));
        }
    }

    void Renderer3D::stroke_bold_polygon2(const Polyloop2 &poly, Color color, float z, float thickness, float alpha)
    {
        vector<Vector3> pts_vector3 = vec2_to_Vector3_arr(poly.points(), float(z));
        size_t n = pts_vector3.size();
        for (size_t i = 0; i < n; ++i)
        {

            DrawCylinderEx(pts_vector3[i], pts_vector3[(i + 1) % n], thickness, thickness, 1, Fade(color, alpha));
        }
    }

    void Renderer3D::stroke_light_polygon3(const Polyloop3 &poly, Color color, float alpha)
    {
        vector<Vector3> pts_vector3 = vec3_to_Vector3_arr(poly.points());
        size_t n = pts_vector3.size();
        for (size_t i = 0; i < n; ++i)
        {
            DrawLine3D(pts_vector3[i], pts_vector3[(i + 1) % n], Fade(color, alpha));
        }
    }

    void Renderer3D::stroke_bold_polygon3(const Polyloop3 &poly, Color color, float thickness, float alpha)
    {
        vector<Vector3> pts_vector3 = vec3_to_Vector3_arr(poly.points());
        size_t n = pts_vector3.size();
        for (size_t i = 0; i < n; ++i)
        {
            DrawCylinderEx(pts_vector3[i], pts_vector3[(i + 1) % n], thickness, thickness, 1, Fade(color, alpha));
        }
    }

    void Renderer3D::draw_light_polyline2(const std::vector<Vec2> &pts, Color color, float z, float alpha)
    {
        vector<Vector3> pts_vector3 = vec2_to_Vector3_arr(pts, float(z));
        size_t n = pts_vector3.size();
        if (n <= 1)
            return;
        for (size_t i = 0; i < n - 1; ++i)
        {
            DrawLine3D(pts_vector3[i], pts_vector3[i + 1], Fade(color, alpha));
        }
    }


    void Renderer3D::draw_bold_polyline2(const std::vector<Vec2> &pts, Color color, float z, float thickness, float alpha)
    {
        vector<Vector3> pts_vector3 = vec2_to_Vector3_arr(pts, float(z));
        size_t n = pts_vector3.size();
        if (n <= 1)
            return;
        for (size_t i = 0; i < n - 1; ++i)
        {
            DrawCylinderEx(pts_vector3[i], pts_vector3[i+1], thickness, thickness, 1, Fade(color, alpha));
        }
    }


    void Renderer3D::draw_light_polyline3(const std::vector<Vec3> &pts, Color color, float alpha)
    {
        vector<Vector3> pts_vector3 = vec3_to_Vector3_arr(pts);
        size_t n = pts_vector3.size();
        if (n <= 1)
            return;
        for (size_t i = 0; i < n - 1; ++i)
        {
            DrawLine3D(pts_vector3[i], pts_vector3[i + 1], Fade(color, alpha));
        }
    }

    void Renderer3D::draw_bold_polyline3(const std::vector<Vec3> &pts, Color color, float thickness, float alpha)
    {
        vector<Vector3> pts_vector3 = vec3_to_Vector3_arr(pts);
        size_t n = pts_vector3.size();
        if (n <= 1)
            return;
        for (size_t i = 0; i < n - 1; ++i)
        {
            DrawCylinderEx(pts_vector3[i], pts_vector3[i + 1], thickness, thickness, 1, Fade(color, alpha));
        }
    }

    void Renderer3D::draw_points(const std::vector<Vec2> &pts, Color color, float alpha, Scalar radius, Scalar z)
    {
        for (auto &pt : pts)
        {
            DrawCube({pt.x(), z, -pt.y()}, radius, radius, radius, Fade(color, alpha));
            // DrawPoint3D({pt.x(), z, -pt.y()}, Fade(color, alpha));
            // DrawSphere({pt.x(), z, -pt.y()}, radius, Fade(color, alpha));
        }
    }

    void Renderer3D::draw_points(const std::vector<Vec3> &pts, Color color, float alpha, Scalar radius)
    {
        for (auto &pt : pts)
        {
            DrawCube({pt.x(), pt.z(), -pt.y()}, radius, radius, radius, Fade(color, alpha));
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
            if (callBack.onDraw3D)
                callBack.onDraw3D();
            EndMode3D();
            if (callBack.onDraw2D)
                callBack.onDraw2D();
            EndDrawing();
        }
    }

    void Renderer3D::draw_index_fonts(const std::vector<Vector3> &world_pos, int size, Color color)
    {
        for (size_t i = 0; i < world_pos.size(); ++i)
        {
            Vector2 screen = GetWorldToScreen(world_pos[i], camera_);

            // 可选：屏幕裁剪
            if (screen.x < 0 || screen.x > GetScreenWidth() ||
                screen.y < 0 || screen.y > GetScreenHeight())
                continue;

            DrawText(
                TextFormat("%zu", i),
                (int)screen.x,
                (int)screen.y,
                size,
                color);
        }
    }

    vector<Vector3> vec2_to_Vector3_arr(const vector<Vec2> &arr2, float z)
    {
        vector<Vector3> arr3;
        arr3.reserve(arr2.size());
        for (const auto &p2 : arr2)
            arr3.push_back(vec2_to_Vector3(p2, z));
        return arr3;
    }

    vector<Vector3> vec3_to_Vector3_arr(const vector<Vec3> &arr)
    {
        vector<Vector3> arr3;
        arr3.reserve(arr.size());
        for (const auto &p2 : arr)
            arr3.push_back(vec3_to_Vector3(p2));
        return arr3;
    }

    void sort_polygon_ccw(std::vector<Vec2> &pts)
    {
        Vec2 center{0, 0};
        for (auto &p : pts)
            center += p;
        center /= float(pts.size());

        std::sort(pts.begin(), pts.end(),
                  [&](const Vec2 &a, const Vec2 &b)
                  {
                      float aa = atan2(a.y() - center.y(), a.x() - center.x());
                      float bb = atan2(b.y() - center.y(), b.x() - center.x());
                      return aa < bb;
                  });
    }

}