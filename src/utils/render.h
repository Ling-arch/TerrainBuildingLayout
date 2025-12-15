#pragma once

#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include <vector>
#include <functional>
#include <earcut.hpp>
#include "voronoi2.h"
#include "util.h"

#define RENDER_MOVE_SPEED 0.03f
#define RENDER_ROTATE_SPEED 0.003f
#define RENDER_ZOOM_SPEED 1.0f

namespace render
{

    using Scalar = voronoi2::Scalar;
    using M2 = util::Math2<Scalar>;
    using Vec2 = typename M2::Vector2;
    using Point2 = std::array<double, 2>;
    using Tri_Point3 = std::array<Vector3, 3>;

    struct Triangle3
    {
        Tri_Point3 points;
        Color color;
        bool doubleSided = false;
    };

    struct RenderPoly{
        std::vector<Triangle3> triangles;
        bool dirty = true;
    };

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

        std::vector<Triangle3> render_triangles;
        std::vector<std::shared_ptr<RenderPoly>> polys;
        void draw_faces();
        void draw_edges();
        void draw_fonts();

    public:
        Renderer3D(int width, int height, float fovy, CameraProjection proj, const char *name);
        ~Renderer3D();

        // 在这个地方进行绘制3d内容
        void begin3D();
        void end3D();
        void updateCamera();
        void clear();
        std::shared_ptr<RenderPoly> createPolyMesh();
        void draw();

        void fill_polygon2(RenderPoly &poly, const std::vector<Vec2> &pts, Scalar z, Color color, float alpha = 1.f,bool doublesided = false);
        void stroke_polygon2(const std::vector<Vec2> &pts, Scalar z, float thickness, Color color, float alpha);

        void fill_polygon3(RenderPoly &poly, const std::vector<Vector3> &pts, Color color, float alpha = 1.f,bool doublesided =false);
        void stroke_polygon3(const std::vector<Vector3> &pts, float thickness, Color color, float alpha);

        void draw_points(const std::vector<Vec2> &pts, Scalar z, Scalar radius, Color color, float alpha);
        void draw_polyline(const std::vector<Vector3> &pts, Color color, float alpha);

        void fill_polymesh(const std::vector<std::vector<Vector3>> &mesh, Color color, float alpha);

        void runMainLoop(const FrameCallbacks &callBack);

        //-----------------------------------辅助方法------------------------------------

        Vector3 vec2_to_Vector3(const Vec2 &v2, float z = 0.f);
        std::vector<Vector3> vec2_to_Vector3(const std::vector<Vec2> &arr2, float z = 0.f);
        bool compute_plane(const std::vector<Vector3> &pts, Vector3 &origin, Vector3 &normal, float eps = 1e-4f);

        void make_plane_basis(const Vector3 &n, Vector3 &u, Vector3 &v);

        std::vector<Vec2> project_to_2d(
            const std::vector<Vector3> &pts,
            const Vector3 &origin,
            const Vector3 &u,
            const Vector3 &v);
    };
}