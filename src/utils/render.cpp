#include "render.h"

#include <iostream>
#include <cmath>

using std::vector;

namespace render
{

    float RENDER_MOVE_SPEED = 0.03f;
    float RENDER_ROTATE_SPEED = 0.003f;
    float RENDER_ZOOM_SPEED = 1.0f;
    float LETTER_BOUNDRY_SIZE = 0.25f;
    int TEXT_MAX_LAYERS = 32;
    Color LETTER_BOUNDRY_COLOR = RL_VIOLET;
    bool SHOW_LETTER_BOUNDRY = false;
    bool SHOW_TEXT_BOUNDRY = false;
    Renderer3D::Renderer3D(int width, int height, float fovy, CameraProjection proj, const char *name)
        : width_(width), height_(height), fovy_(fovy), projection_(proj), name_(name),
          yaw_(0.8f), pitch_(0.4f), distance_(14.f)
    {
        SetConfigFlags(FLAG_MSAA_4X_HINT);

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

        /* ================= Rotate ================= */
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
        {
            yaw_ += md.x * RENDER_ROTATE_SPEED;
            pitch_ += md.y * RENDER_ROTATE_SPEED;

            float limit = PI / 2.f - 0.01f;
            pitch_ = Clamp(pitch_, -limit, limit);
        }

        /* ================= Zoom ================= */
        float wheel = GetMouseWheelMove();
        if (wheel != 0.f)
        {
            if (camera_.projection == CAMERA_ORTHOGRAPHIC)
            {
                // Orthographic：改可视范围
                camera_.fovy -= wheel * RENDER_ZOOM_SPEED;
                camera_.fovy = Clamp(camera_.fovy, 0.1f, 10000.f);
            }
            else
            {
                // Perspective：改距离
                distance_ -= wheel * RENDER_ZOOM_SPEED;
                distance_ = std::max(distance_, 0.1f);
            }
        }

        /* ================= Pan ================= */
        if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))
        {
            Vector3 forward = {
                cosf(pitch_) * sinf(yaw_),
                sinf(pitch_),
                cosf(pitch_) * cosf(yaw_)};

            Vector3 right, up;

            // 是否接近 Top / Bottom
            const float verticalLimit = 0.99f;
            if (currentView == RenderView::Top || currentView == RenderView::Bottom)
            {
                // ✅ Top / Bottom：强制世界轴
                right = {1.f, 0.f, 0.f}; // X+
                up = {0.f, 1.f, 0.f};    // Y+（关键修正点）
            }
            else
            {
                Vector3 worldUp = {0.f, 1.f, 0.f};
                right = Vector3Normalize(Vector3CrossProduct(forward, worldUp));
                up = Vector3Normalize(Vector3CrossProduct(right, forward));
            }

            float scale = RENDER_MOVE_SPEED;
            Vector3 pan;
            // Orthographic 下拖拽要乘以视口大小
            if (camera_.projection == CAMERA_ORTHOGRAPHIC)
            {
                scale *= camera_.fovy * 0.01f;
                pan = Vector3Add(
                    Vector3Scale(right, md.x * scale),
                    Vector3Scale(up, md.y * scale));
            }
            else
            {
                pan = Vector3Add(
                    Vector3Scale(right, md.x * scale),
                    Vector3Scale(up, md.y * scale));
            }

            camera_.target = Vector3Add(camera_.target, pan);
        }

        /* ================= Position ================= */
        Vector3 offset = {
            distance_ * cosf(pitch_) * sinf(yaw_),
            distance_ * sinf(pitch_),
            distance_ * cosf(pitch_) * cosf(yaw_)};

        camera_.position = Vector3Add(camera_.target, offset);
        camera_.up = {0.f, 1.f, 0.f};
    }

    void fill_polygon2(const Polyloop2 &poly, Color color, float z, float alpha, Vec2 move, bool doubleSided)
    {
        if (poly.triangles().size() <= 0)
            return;
        for (auto &tri : poly.triangles())
        {
            Vec2 a = poly.points()[tri.at(0)] + move;
            Vec2 b = poly.points()[tri.at(1)] + move;
            Vec2 c = poly.points()[tri.at(2)] + move;
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

    void fill_polygon3(const Polyloop3 &poly, Color color, float alpha, Vec3 move, bool doubleSided)
    {
        if (poly.triangles().size() <= 0)
            return;
        for (auto &tri : poly.triangles())
        {
            Vec3 a = poly.points()[tri.at(0)] + move;
            Vec3 b = poly.points()[tri.at(1)] + move;
            Vec3 c = poly.points()[tri.at(2)] + move;
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

    void stroke_light_polygon2(const Polyloop2 &poly, Color color, float z, float alpha, Vec2 move)
    {
        vector<Vector3> pts_vector3 = vec2_to_Vector3_arr(poly.points(), float(z));
        size_t n = pts_vector3.size();
        if (n <= 1)
            return;
        Vector3 delta = {move.x(), 0, -move.y()};
        for (size_t i = 0; i < n; ++i)
        {
            DrawLine3D(pts_vector3[i] + delta, pts_vector3[(i + 1) % n] + delta, Fade(color, alpha));
        }
    }

    void stroke_bold_polygon2(const Polyloop2 &poly, Color color, float z, float thickness, float alpha, Vec2 move)
    {
        vector<Vector3> pts_vector3 = vec2_to_Vector3_arr(poly.points(), float(z));
        size_t n = pts_vector3.size();
        if (n <= 1)
            return;
        Vector3 delta = {move.x(), 0, -move.y()};
        for (size_t i = 0; i < n; ++i)
        {
            DrawCylinderEx(pts_vector3[i] + delta, pts_vector3[(i + 1) % n] + delta, thickness, thickness, 1, Fade(color, alpha));
        }
    }

    void stroke_light_polygon3(const Polyloop3 &poly, Color color, float alpha, Vec3 move)
    {
        vector<Vector3> pts_vector3 = vec3_to_Vector3_arr(poly.points());
        size_t n = pts_vector3.size();
        if (n <= 1)
            return;
        Vector3 delta = {move.x(), move.z(), -move.y()};
        for (size_t i = 0; i < n; ++i)
        {
            DrawLine3D(pts_vector3[i] + delta, pts_vector3[(i + 1) % n] + delta, Fade(color, alpha));
        }
    }

    void stroke_bold_polygon3(const Polyloop3 &poly, Color color, float thickness, float alpha, Vec3 move)
    {
        vector<Vector3> pts_vector3 = vec3_to_Vector3_arr(poly.points());
        size_t n = pts_vector3.size();
        if (n <= 1)
            return;
        Vector3 delta = {move.x(), move.z(), -move.y()};
        for (size_t i = 0; i < n; ++i)
        {
            DrawCylinderEx(pts_vector3[i] + delta, pts_vector3[(i + 1) % n] + delta, thickness, thickness, 1, Fade(color, alpha));
        }
    }

    void draw_light_polyline2(const std::vector<Vec2> &pts, Color color, float z, float alpha, Vec2 move)
    {
        size_t n = pts.size();
        if (n <= 1)
            return;
        vector<Vector3> pts_vector3 = vec2_to_Vector3_arr(pts, float(z));
        Vector3 delta = {move.x(), 0, -move.y()};
        for (size_t i = 0; i < n - 1; ++i)
        {
            DrawLine3D(pts_vector3[i] + delta, pts_vector3[i + 1] + delta, Fade(color, alpha));
        }
    }

    void draw_bold_polyline2(const std::vector<Vec2> &pts, Color color, float z, float thickness, float alpha, Vec2 move)
    {
        size_t n = pts.size();
        if (n <= 1)
            return;
        vector<Vector3> pts_vector3 = vec2_to_Vector3_arr(pts, float(z));
        Vector3 delta = {move.x(), 0, -move.y()};
        for (size_t i = 0; i < n - 1; ++i)
        {
            DrawCylinderEx(pts_vector3[i] + delta, pts_vector3[i + 1] + delta, thickness, thickness, 1, Fade(color, alpha));
        }
    }

    void draw_light_polyline3(const std::vector<Vec3> &pts, Color color, float alpha, Vec3 move)
    {

        size_t n = pts.size();
        if (n <= 1)
            return;
        vector<Vector3> pts_vector3 = vec3_to_Vector3_arr(pts);
        Vector3 delta = {move.x(), move.z(), -move.y()};
        for (size_t i = 0; i < n - 1; ++i)
        {
            DrawLine3D(pts_vector3[i] + delta, pts_vector3[i + 1] + delta, Fade(color, alpha));
        }
    }

    void draw_bold_polyline3(const std::vector<Vec3> &pts, Color color, float thickness, float alpha, Vec3 move)
    {
        size_t n = pts.size();
        if (n <= 1)
            return;
        vector<Vector3> pts_vector3 = vec3_to_Vector3_arr(pts);
        Vector3 delta = {move.x(), move.z(), -move.y()};
        for (size_t i = 0; i < n - 1; ++i)
        {
            DrawCylinderEx(pts_vector3[i] + delta, pts_vector3[i + 1] + delta, thickness, thickness, 1, Fade(color, alpha));
        }
    }

    void draw_points(const std::vector<Vec2> &pts, Color color, float alpha, Scalar radius, Scalar z, Vec2 move)
    {
        if (pts.size() < 0)
            return;
        for (auto &pt : pts)
        {
            DrawCube({pt.x() + move.x(), z, -pt.y() - move.y()}, radius, radius, radius, Fade(color, alpha));
            // DrawPoint3D({pt.x(), z, -pt.y()}, Fade(color, alpha));
            // DrawSphere({pt.x(), z, -pt.y()}, radius, Fade(color, alpha));
        }
    }

    void draw_points(const std::vector<Vec3> &pts, Color color, float alpha, Scalar radius, Vec3 move)
    {
        if (pts.size() < 0)
            return;
        for (auto &pt : pts)
        {
            DrawCube({pt.x() + move.x(), pt.z() + move.z(), -pt.y() - move.y()}, radius, radius, radius, Fade(color, alpha));
        }
    }

    void draw_vector(const Vec3 &start, const Vec3 &dir, Color color, float scale, float startThickness, float endThickness, float alpha, Vec3 move)
    {
        Vec3 strat_new = start + move;
        Vec3 end = strat_new + dir * scale;
        DrawCylinderEx(
            {strat_new.x(), strat_new.z(), -strat_new.y()},
            {end.x(), end.z(), -end.y()},
            startThickness,
            endThickness,
            1,
            Fade(color, alpha));
    }

    void draw_vector(const Vec2 &start, const Vec2 &dir, Color color, float scale, float startThickness, float endThickness, float z, float alpha, Vec2 move)
    {
        Vec2 strat_new = start + move;
        Vec2 end = strat_new + dir * scale;
        DrawCylinderEx(
            {strat_new.x(), z, -strat_new.y()},
            {end.x(), z, -end.y()},
            startThickness,
            endThickness,
            1,
            Fade(color, alpha));
    }

    void DrawTextCodepoint3D(Font font, int codepoint, Vector3 position, float fontSize, bool backface, Color tint)
    {
        // Character index position in sprite font
        // NOTE: In case a codepoint is not available in the font, index returned points to '?'
        int index = GetGlyphIndex(font, codepoint);
        float scale = fontSize / (float)font.baseSize;

        // Character destination rectangle on screen
        // NOTE: We consider charsPadding on drawing
        position.x += (float)(font.glyphs[index].offsetX - font.glyphPadding) * scale;
        position.z += (float)(font.glyphs[index].offsetY - font.glyphPadding) * scale;

        // Character source rectangle from font texture atlas
        // NOTE: We consider chars padding when drawing, it could be required for outline/glow shader effects
        Rectangle srcRec = {font.recs[index].x - (float)font.glyphPadding, font.recs[index].y - (float)font.glyphPadding,
                            font.recs[index].width + 2.0f * font.glyphPadding, font.recs[index].height + 2.0f * font.glyphPadding};

        float width = (float)(font.recs[index].width + 2.0f * font.glyphPadding) * scale;
        float height = (float)(font.recs[index].height + 2.0f * font.glyphPadding) * scale;

        if (font.texture.id > 0)
        {
            const float x = 0.0f;
            const float y = 0.0f;
            const float z = 0.0f;

            // normalized texture coordinates of the glyph inside the font texture (0.0f -> 1.0f)
            const float tx = srcRec.x / font.texture.width;
            const float ty = srcRec.y / font.texture.height;
            const float tw = (srcRec.x + srcRec.width) / font.texture.width;
            const float th = (srcRec.y + srcRec.height) / font.texture.height;

            if (SHOW_LETTER_BOUNDRY)
                DrawCubeWiresV({position.x + width / 2, position.y, position.z + height / 2}, {width, LETTER_BOUNDRY_SIZE, height}, LETTER_BOUNDRY_COLOR);

            rlCheckRenderBatchLimit(4 + 4 * backface);
            rlSetTexture(font.texture.id);

            rlPushMatrix();
            rlTranslatef(position.x, position.y, position.z);

            rlBegin(RL_QUADS);
            rlColor4ub(tint.r, tint.g, tint.b, tint.a);

            // Front Face
            rlNormal3f(0.0f, 1.0f, 0.0f); // Normal Pointing Up
            rlTexCoord2f(tx, ty);
            rlVertex3f(x, y, z); // Top Left Of The Texture and Quad
            rlTexCoord2f(tx, th);
            rlVertex3f(x, y, z + height); // Bottom Left Of The Texture and Quad
            rlTexCoord2f(tw, th);
            rlVertex3f(x + width, y, z + height); // Bottom Right Of The Texture and Quad
            rlTexCoord2f(tw, ty);
            rlVertex3f(x + width, y, z); // Top Right Of The Texture and Quad

            if (backface)
            {
                // Back Face
                rlNormal3f(0.0f, -1.0f, 0.0f); // Normal Pointing Down
                rlTexCoord2f(tx, ty);
                rlVertex3f(x, y, z); // Top Right Of The Texture and Quad
                rlTexCoord2f(tw, ty);
                rlVertex3f(x + width, y, z); // Top Left Of The Texture and Quad
                rlTexCoord2f(tw, th);
                rlVertex3f(x + width, y, z + height); // Bottom Left Of The Texture and Quad
                rlTexCoord2f(tx, th);
                rlVertex3f(x, y, z + height); // Bottom Right Of The Texture and Quad
            }
            rlEnd();
            rlPopMatrix();

            rlSetTexture(0);
        }
    }

    void DrawText3D(Font font, const char *text, Vector3 position, float fontSize, float fontSpacing, float lineSpacing, bool backface, Color tint)
    {
        int length = TextLength(text); // Total length in bytes of the text, scanned by codepoints in loop

        float textOffsetY = 0.0f; // Offset between lines (on line break '\n')
        float textOffsetX = 0.0f; // Offset X to next character to draw

        float scale = fontSize / (float)font.baseSize;

        for (int i = 0; i < length;)
        {
            // Get next codepoint from byte string and glyph index in font
            int codepointByteCount = 0;
            int codepoint = GetCodepoint(&text[i], &codepointByteCount);
            int index = GetGlyphIndex(font, codepoint);

            // NOTE: Normally we exit the decoding sequence as soon as a bad byte is found (and return 0x3f)
            // but we need to draw all of the bad bytes using the '?' symbol moving one byte
            if (codepoint == 0x3f)
                codepointByteCount = 1;

            if (codepoint == '\n')
            {
                // NOTE: Fixed line spacing of 1.5 line-height
                // TODO: Support custom line spacing defined by user
                textOffsetY += fontSize + lineSpacing;
                textOffsetX = 0.0f;
            }
            else
            {
                if ((codepoint != ' ') && (codepoint != '\t'))
                {
                    DrawTextCodepoint3D(font, codepoint, {position.x + textOffsetX, position.y, position.z + textOffsetY}, fontSize, backface, tint);
                }

                if (font.glyphs[index].advanceX == 0)
                    textOffsetX += (float)font.recs[index].width * scale + fontSpacing;
                else
                    textOffsetX += (float)font.glyphs[index].advanceX * scale + fontSpacing;
            }

            i += codepointByteCount; // Move text bytes counter to next codepoint
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
            ClearBackground(RL_WHITE);
            rlSetBlendMode(RL_BLEND_ALPHA);
            BeginMode3D(camera_);
            if (callBack.onDraw3D)
                callBack.onDraw3D();
            EndMode3D();
            if (callBack.onDraw2D)
                callBack.onDraw2D();
            EndDrawing();
        }
    }

    void Renderer3D::setCameraUI(bool customOpen)
    {
        if (!ImGui::Begin("Render Settings", &customOpen))
        {
            ImGui::End();
            return;
        }

        /* ================= Camera Control ================= */
        if (ImGui::CollapsingHeader("Camera Control", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderFloat("Move Speed", &render::RENDER_MOVE_SPEED, 0.f, 2.f, "%.2f");
            ImGui::SliderFloat("Rotate Speed", &render::RENDER_ROTATE_SPEED, 0.f, 0.01f, "%.4f");
            ImGui::SliderFloat("Zoom Speed", &render::RENDER_ZOOM_SPEED, 0.f, 10.f, "%.2f");
        }

        /* ================= View Presets ================= */
        if (ImGui::CollapsingHeader("View Presets", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Top"))
            {
                currentView = RenderView::Top;
                yaw_ = 0.f;
                pitch_ = +PI / 2.f - 0.001f; // ✅ 从 Y+ 往下看
                camera_.projection = CAMERA_ORTHOGRAPHIC;
            }
            ImGui::SameLine();

            if (ImGui::Button("Bottom"))
            {
                currentView = RenderView::Bottom;
                yaw_ = 0.f;
                pitch_ = -PI / 2.f + 0.001f;
                camera_.projection = CAMERA_ORTHOGRAPHIC;
            }
            ImGui::SameLine();

            if (ImGui::Button("Front"))
            {
                currentView = RenderView::Front;
                yaw_ = 0.f;
                pitch_ = 0.f;
                camera_.projection = CAMERA_ORTHOGRAPHIC;
            }
            ImGui::SameLine();

            if (ImGui::Button("Back"))
            {
                currentView = RenderView::Back;
                yaw_ = PI;
                pitch_ = 0.f;
                camera_.projection = CAMERA_ORTHOGRAPHIC;
            }
            ImGui::SameLine();

            if (ImGui::Button("Left"))
            {
                currentView = RenderView::Left;
                yaw_ = -PI / 2.f;
                pitch_ = 0.f;
                camera_.projection = CAMERA_ORTHOGRAPHIC;
            }
            ImGui::SameLine();

            if (ImGui::Button("Right"))
            {
                currentView = RenderView::Right;
                yaw_ = PI / 2.f;
                pitch_ = 0.f;
                camera_.projection = CAMERA_ORTHOGRAPHIC;
            }

            if (ImGui::Button("Isometric"))
            {
                currentView = RenderView::Free;
                yaw_ = PI / 4.f;
                pitch_ = +35.264f * DEG2RAD; // 注意这里是正的（从上往下）
                camera_.projection = CAMERA_PERSPECTIVE;
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset"))
            {
                currentView = RenderView::Free;
                yaw_ = 0.8f;
                pitch_ = 0.4f;
                distance_ = 14.f;
                // camera_.target = {0.f, 0.f, 0.f};
                camera_.projection = CAMERA_PERSPECTIVE;
            }

            // 正交视图下的 CAD 风格旋转
            if (currentView != RenderView::Free)
            {
                if (ImGui::Button("Rotate 90° CCW"))
                    yaw_ += PI / 2.f;
                ImGui::SameLine();
                if (ImGui::Button("Rotate 90° CW"))
                    yaw_ -= PI / 2.f;
            }
        }

        /* ================= Projection ================= */
        if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen))
        {
            int proj = (camera_.projection == CAMERA_PERSPECTIVE) ? 0 : 1;

            if (ImGui::RadioButton("Perspective", proj == 0))
            {
                camera_.projection = CAMERA_PERSPECTIVE;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Orthographic", proj == 1))
            {
                camera_.projection = CAMERA_ORTHOGRAPHIC;
            }

            if (camera_.projection == CAMERA_PERSPECTIVE)
            {
                ImGui::SliderFloat("FOV", &camera_.fovy, 10.f, 120.f, "%.1f");
            }
            else
            {
                ImGui::SliderFloat("Ortho Size", &camera_.fovy, 1.f, 100.f, "%.1f");
            }
        }

        /* ================= Camera Transform ================= */
        if (ImGui::CollapsingHeader("Camera Transform", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::DragFloat("Yaw", &yaw_, 0.01f, -PI, PI, "%.3f");
            ImGui::DragFloat("Pitch", &pitch_, 0.01f, -PI / 2.f + 0.01f, PI / 2.f - 0.01f, "%.3f");
            ImGui::SliderFloat("Distance", &distance_, 0.1f, 100000.f, "%.2f");

            ImGui::Separator();
            ImGui::Text("Target");
            ImGui::DragFloat3("Target Pos", &camera_.target.x, 0.1f);
        }

        ImGui::End();
    }

    void Renderer3D::setDrawGeoDataUI(bool customOpen)
    {
        if (ImGui::Begin("Geo Settings", &customOpen))
        {
            if (ImGui::TreeNodeEx("Point Settings", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat("Point Size", &ptData.size, 0.01f, 10.f, "%.2f");
                if (ImGui::ColorEdit4("Point Color", (float *)&ptData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                    ptData.syncFloatToColor();
                ImGui::TreePop();
            }
            if (ImGui::TreeNodeEx("Vector Settings", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat("Vector Scale", &vecData.scale, 1.f, 20.f, "%.1f");
                ImGui::SliderFloat("Start Thickness", &vecData.startThickness, 0.1f, 10.f, "%.1f");
                ImGui::SliderFloat("End Thickness", &vecData.endThickness, 0.1f, 10.f, "%.1f");
                ImGui::SliderFloat("Vec Z", &vecData.vecZ, -100.0f, 100.f, "%.1f");
                if (ImGui::ColorEdit4("Vector Color", (float *)&vecData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                    vecData.syncFloatToColor();
                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Font Settings", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderInt("Font Size", &fontData.size, 1, 30);
                if (ImGui::ColorEdit4("Font Color", (float *)&fontData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                    fontData.syncFloatToColor();
                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Line Settings", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat("Line Thickness", &lineData.Thickness, 0.01f, 1.f, "%.2f");
                if (ImGui::ColorEdit4("Line Color", (float *)&lineData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                    lineData.syncFloatToColor();
                ImGui::TreePop();
            }
        }
        ImGui::End();
    }

    void Renderer3D::draw_index_fonts(const std::vector<Vector3> &world_pos, int size, Color color)
    {
        if (world_pos.empty())
            return;
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

    void Renderer3D::draw_index_fonts(const std::vector<Vec3> &world_pos, int size, Color color, Vec3 move)
    {
        if (world_pos.empty())
            return;
        for (size_t i = 0; i < world_pos.size(); ++i)
        {
            Vector2 screen = GetWorldToScreen(vec3_to_Vector3(world_pos[i] + move), camera_);

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

    void Renderer3D::draw_index_fonts(const std::vector<Vec2> &world_pos, int size, Color color, float z, Vec2 move)
    {
        if(world_pos.empty())
            return;
        for (size_t i = 0; i < world_pos.size(); ++i)
        {
            Vector2 screen = GetWorldToScreen(vec2_to_Vector3(world_pos[i] + move, z), camera_);

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