#pragma once

#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include <vector>
#include <functional>
#include <earcut.hpp>
#include "voronoi2.h"
#include "util.h"
#include "polyloop.h"
#include <rlImGui.h>
#include <imgui.h>

#define RENDER_MOVE_SPEED 0.03f
#define RENDER_ROTATE_SPEED 0.003f
#define RENDER_ZOOM_SPEED 1.0f

namespace render
{
    using Scalar = voronoi2::Scalar;
    using M2 = util::Math2<Scalar>;
    using Vec2 = typename M2::Vector2;
    using Vec3 = typename M2::Vector3;
    using Point2 = std::array<double, 2>;
    using Tri_Point3 = std::array<Vector3, 3>;
    using polyloop::Polyloop2, polyloop::Polyloop3;

    struct FrameCallbacks
    {
        std::function<void()> onUpdate;
        std::function<void()> onDraw3D;
        std::function<void()> onDraw2D;
    };

    class Renderer3D
    {
    private:
        const char *name_;
        int width_, height_;
        float fovy_;
        CameraProjection projection_;
        Camera camera_;
        float yaw_, pitch_, distance_;
        bool in3D = false;
        bool drawBack = false;

    public:
        Renderer3D(int width, int height, float fovy, CameraProjection proj, const char *name);
        ~Renderer3D();

        void updateCamera();

        void draw();
        void fill_polygon2(const Polyloop2 &poly, Color color, float z = 0, float alpha = 1.f, bool doubleSided = false);
        void fill_polygon3(const Polyloop3 &poly, Color color, float alpha = 1.f, bool doubleSided = false);

        void stroke_light_polygon2(const Polyloop2 &poly, Color color, float z = 0.f, float alpha = 1.f);
        void stroke_bold_polygon2(const Polyloop2 &poly, Color color, float z = 0.f, float thickness = 0.03f, float alpha = 1.f);
        void stroke_light_polygon3(const Polyloop3 &poly, Color color, float alpha = 1.f);
        void stroke_bold_polygon3(const Polyloop3 &poly, Color color, float thickness = 0.03f, float alpha = 1.f);

        void draw_points(const std::vector<Vec2> &pts, Color color, float alpha = 1.f, Scalar radius = 0.1f, Scalar z = 0.f);
        void draw_points(const std::vector<Vec3> &pts, Color color, float alpha = 1.f, Scalar radius = 0.1f);

        void draw_light_polyline2(const std::vector<Vec2> &pts, Color color, float z = 0.f, float alpha = 1.f);
        void draw_bold_polyline2(const std::vector<Vec2> &pts, Color color, float z = 0.f,float thickness = 0.03f,float alpha = 1.f);
        void draw_light_polyline3(const std::vector<Vec3> &pts, Color color, float alpha = 1.f);
        void draw_bold_polyline3(const std::vector<Vec3> &pts, Color color, float thickness = 0.03f, float alpha = 1.f);

        void draw_index_fonts(const std::vector<Vector3> &world_pos, int size, Color color);

        void fill_polymesh(const std::vector<std::vector<Vector3>> &mesh, Color color, float alpha = 1.f);

        void runMainLoop(const FrameCallbacks &callBack);
    };

    //-----------------------------------辅助方法------------------------------------

    inline Vector3 vec2_to_Vector3(const Vec2 &v2, float z)
    {
        return {v2.x(), z, -v2.y()};
    }

    inline Vector3 vec3_to_Vector3(const Vec3 &v3)
    {
        return {v3.x(), v3.z(), -v3.y()};
    }

    std::vector<Vector3> vec3_to_Vector3_arr(const std::vector<Vec3> &arr);
    std::vector<Vector3> vec2_to_Vector3_arr(const std::vector<Vec2> &arr2, float z = 0.f);

    void sort_polygon_ccw(std::vector<Vec2> &pts);
}