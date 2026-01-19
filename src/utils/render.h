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

namespace render
{
    using Scalar = voronoi2::Scalar;
    using M2 = util::Math2<Scalar>;
    using Vec2 = typename M2::Vector2;
    using Vec3 = typename M2::Vector3;
    using Point2 = std::array<double, 2>;
    using Tri_Point3 = std::array<Vector3, 3>;
    using polyloop::Polyloop2, polyloop::Polyloop3;

    extern float RENDER_MOVE_SPEED ;
    extern float RENDER_ROTATE_SPEED ;
    extern float RENDER_ZOOM_SPEED;

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
        Camera& getCamera() { return camera_; }
        void updateCamera();
        void draw_index_fonts(const std::vector<Vector3> &world_pos, int size, Color color);
        void draw_index_fonts(const std::vector<Vec3> &world_pos, int size, Color color,Vec3 move = {0.f, 0.f, 0.f});
        void draw_index_fonts(const std::vector<Vec2> &world_pos, int size, Color color,float z = 0.f,Vec2 move = {0.f, 0.f});
        void runMainLoop(const FrameCallbacks &callBack);
    };

    //-----------------------------------辅助方法------------------------------------

    void fill_polygon2(const Polyloop2 &poly, Color color, float z = 0, float alpha = 1.f , Vec2 move = {0.f, 0.f} ,bool doubleSided = false);
    void fill_polygon3(const Polyloop3 &poly, Color color, float alpha = 1.f, Vec3 move = {0.f, 0.f, 0.f} ,bool doubleSided = false);

    void stroke_light_polygon2(const Polyloop2 &poly, Color color, float z = 0.f, float alpha = 1.f, Vec2 move = {0.f, 0.f});
    void stroke_bold_polygon2(const Polyloop2 &poly, Color color, float z = 0.f, float thickness = 0.03f, float alpha = 1.f,Vec2 move = {0.f,0.f});
    void stroke_light_polygon3(const Polyloop3 &poly, Color color, float alpha = 1.f, Vec3 move = {0.f, 0.f, 0.f});
    void stroke_bold_polygon3(const Polyloop3 &poly, Color color, float thickness = 0.03f, float alpha = 1.f, Vec3 move = {0.f, 0.f, 0.f});

    void draw_points(const std::vector<Vec2> &pts, Color color, float alpha = 1.f, Scalar radius = 0.1f, Scalar z = 0.f, Vec2 move = {0.f, 0.f});
    void draw_points(const std::vector<Vec3> &pts, Color color, float alpha = 1.f, Scalar radius = 0.1f, Vec3 move = {0.f, 0.f, 0.f});

    void draw_light_polyline2(const std::vector<Vec2> &pts, Color color, float z = 0.f, float alpha = 1.f, Vec2 move = {0.f, 0.f});
    void draw_bold_polyline2(const std::vector<Vec2> &pts, Color color, float z = 0.f, float thickness = 0.03f, float alpha = 1.f, Vec2 move = {0.f, 0.f});
    void draw_light_polyline3(const std::vector<Vec3> &pts, Color color, float alpha = 1.f, Vec3 move = {0.f, 0.f, 0.f});
    void draw_bold_polyline3(const std::vector<Vec3> &pts, Color color, float thickness = 0.03f, float alpha = 1.f, Vec3 move = {0.f, 0.f, 0.f});

    void fill_polymesh(const std::vector<std::vector<Vector3>> &mesh, Color color, float alpha = 1.f);

    void draw_vector(const Vec3 &start, const Vec3 &dir, Color color, float scale = 1.f, float startThickness = 0.01f, float endThickness = 0.05f, float alpha = 1.f, Vec3 move = {0.f, 0.f, 0.f});
    void draw_vector(const Vec2 &start, const Vec2 &dir, Color color, float scale = 1.f, float startThickness = 0.01f, float endThickness = 0.05f, float z = 0.f, float alpha = 1.f, Vec2 move = {0.f, 0.f});
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

    struct PointDrawData
    {
        Color color = RL_BLACK;
        float colorF[4] = {0.f, 0.f, 0.f, 1.f};
        float size = 0.2f;
        

        PointDrawData() {
            syncFloatToColor();
        };
        PointDrawData(Color c, float s) : color(c), size(s) {}
        void syncFloatToColor()
        {
            color.r = static_cast<unsigned char>(colorF[0] * 255);
            color.g = static_cast<unsigned char>(colorF[1] * 255);
            color.b = static_cast<unsigned char>(colorF[2] * 255);
            color.a = static_cast<unsigned char>(colorF[3] * 255);
          
        }
    };


    struct VectorDrawData
    {
        Color color = RL_BLACK;
        float colorF[4] = {0.f, 0.f, 0.f, 1.f};
        float scale = 1.f;
        float startThickness = 0.02f;
        float endThickness = 0.05f;

        VectorDrawData() {
            syncFloatToColor();
        };
        VectorDrawData(Color c, float s, float startT, float endT) : color(c), scale(s), startThickness(startT), endThickness(endT) {}
        void syncFloatToColor()
        {
            color.r = static_cast<unsigned char>(colorF[0] * 255);
            color.g = static_cast<unsigned char>(colorF[1] * 255);
            color.b = static_cast<unsigned char>(colorF[2] * 255);
            color.a = static_cast<unsigned char>(colorF[3] * 255);
        }
    };


    struct FontDrawData{
        Color color = RL_BLACK;
        float colorF[4] = {0.f, 0.f, 0.f, 1.f};
        int size = 12;

        FontDrawData() {
            syncFloatToColor();
        };
        FontDrawData(Color c, int s) : color(c), size(s) {}
        void syncFloatToColor()
        {
            color.r = static_cast<unsigned char>(colorF[0] * 255);
            color.g = static_cast<unsigned char>(colorF[1] * 255);
            color.b = static_cast<unsigned char>(colorF[2] * 255);
            color.a = static_cast<unsigned char>(colorF[3] * 255);
        }
    };

    struct LineDrawData{
        Color color = RL_BLACK;
        float colorF[4] = {0.f, 0.f, 0.f, 1.f};
        float Thickness = 0.03f;

        LineDrawData() {
            syncFloatToColor();
        };
        LineDrawData(Color c, float t) : color(c), Thickness(t) {}
        void syncFloatToColor()
        {
            color.r = static_cast<unsigned char>(colorF[0] * 255);
            color.g = static_cast<unsigned char>(colorF[1] * 255);
            color.b = static_cast<unsigned char>(colorF[2] * 255);
            color.a = static_cast<unsigned char>(colorF[3] * 255);
        }
    };


}