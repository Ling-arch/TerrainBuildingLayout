#include "tensorField.h"
#include "render.h"

using Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector2d;
using field::Polyline2_t, field::TensorField2D, field::PointAtrractor;
using namespace render;
using M2 = util::Math2<float>;

bool ScreenToWorldXZ(
    const Camera3D &cam,
    Vector2 screenPos,
    Vector3 &outWorldPos,
    Vector2f &outWorldPos2D)
{
    Ray ray = GetMouseRay(screenPos, cam);

    // 平面 y = 0
    float t = -ray.position.y / ray.direction.y;
    if (t < 0)
        return false;

    outWorldPos = {
        ray.position.x + ray.direction.x * t,
        0.0f,
        ray.position.z + ray.direction.z * t};
    outWorldPos2D = Vector2f(outWorldPos.x, -outWorldPos.z);
    return true;
}

int main()
{
    // rlGetCullDistanceNear();
    // rlGetCullDistanceFar();
    rlSetClipPlanes(0.01, 100000.0);
    Renderer3D render(1920, 1080, 45.0f, CAMERA_PERSPECTIVE, "TensorFieldTest");
    render::RENDER_ZOOM_SPEED = 8.f;
    render::RENDER_MOVE_SPEED = 1.f;
    static int ptNum = 5;
    static float scale = 300.f;
    static float threshold = 0.5f;
    static float radius = 50.f;
    PointDrawData ptData;
    VectorDrawData vecData;
    FontDrawData fontData;
    LineDrawData lineData;
    bool showIndices = true;
    bool genPlot = false;
    bool radiusChanged = false;
    Polyline2_t<float> poly = field::createRandomPolygon(ptNum, scale, threshold, Vector2f(0.f, 0.f));
    std::vector<PointAtrractor<float>> attractors;
    attractors.emplace_back(Vector2f(0.f, 0.f), radius);
    TensorField2D<float> tensorField(field::computeAABB<float>({poly}), 20, {poly}, attractors);
    std::vector<Vector2f> seedPoints = M2::gen_poisson_sites_in_poly(poly.points, 1.2f * tensorField.getGridSize(), 30, (unsigned)time(nullptr));
    std::vector<Polyline2_t<float>> streamlines = tensorField.genStreamlines(seedPoints);
    bool hasTestTensor = false;
    field::Tensor<float> testTensor;
    rlImGuiSetup(true);

    render.runMainLoop(render::FrameCallbacks{
        [&]() { // 按键更新，重新绘图等事件
            if (genPlot)
            {
                poly = field::createRandomPolygon(ptNum, scale, threshold, Vector2f(0.f, 0.f));
                tensorField = TensorField2D<float>(field::computeAABB<float>({poly}), 20, {poly}, attractors);
                seedPoints = M2::gen_poisson_sites_in_poly(poly.points, 1.2f * tensorField.getGridSize(), 30, (unsigned)time(nullptr));
                streamlines = tensorField.genStreamlines(seedPoints);
                genPlot = false;
            }

            if (radiusChanged)
            {
                for(auto &attr : attractors)
                    attr.radius = radius;
                tensorField = TensorField2D<float>(field::computeAABB<float>({poly}), 20, {poly}, attractors);
                streamlines = tensorField.genStreamlines(seedPoints);
                radiusChanged = false;
            }

            if (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
            {
                Vector2 mouse = GetMousePosition();
                Vector3 worldPos;
                Vector2f outWorldPos2D;

                if (ScreenToWorldXZ(render.getCamera(), mouse, worldPos, outWorldPos2D))
                {
                    testTensor = tensorField.testTensorAt(outWorldPos2D);
                    hasTestTensor = true;

                    std::cout << "Test pos: "
                              << outWorldPos2D.x() << ", "
                              << outWorldPos2D.y() << std::endl;
                }
            }

            if (IsKeyDown(KEY_LEFT_ALT) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
            {
                Vector2 mouse = GetMousePosition();
                Vector3 worldPos;
                Vector2f outWorldPos2D;

                if (ScreenToWorldXZ(render.getCamera(), mouse, worldPos, outWorldPos2D))
                {
                    attractors.emplace_back(outWorldPos2D, radius);
                    tensorField = TensorField2D<float>(field::computeAABB<float>({poly}), 20, {poly}, attractors);
                    streamlines = tensorField.genStreamlines(seedPoints);
                }
            }

            if (IsKeyDown(KEY_LEFT_SHIFT) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
            {
                Vector2 mouse = GetMousePosition();
                Vector3 worldPos;
                Vector2f outWorldPos2D;

                if (ScreenToWorldXZ(render.getCamera(), mouse, worldPos, outWorldPos2D))
                {
                    float minDist2 = std::numeric_limits<float>::max();
                    int removeIdx = -1;
                    for(const auto &attr : attractors)
                    {
                        float dist2 = (attr.pos - outWorldPos2D).squaredNorm();
                        if(dist2 < minDist2)
                        {
                            minDist2 = dist2;
                            removeIdx = &attr - &attractors[0];
                        }
                    }
                    if(removeIdx != -1)
                        attractors.erase(attractors.begin() + removeIdx);
                    tensorField = TensorField2D<float>(field::computeAABB<float>({poly}), 20, {poly}, attractors);
                    streamlines = tensorField.genStreamlines(seedPoints);
                }
            }


        },
        [&]() { // 3维空间绘图内容部分
            DrawLine3D({0, 0, 0}, {10000, 0, 0}, RED);
            DrawLine3D({0, 0, 0}, {0, 10000, 0}, BLUE);
            DrawLine3D({0, 0, 0}, {0, 0, -10000}, GREEN);
            // DrawGrid(100, 100.f);
            render::stroke_bold_polygon2(polyloop::Polyloop2(poly.points), BLACK, 0.f, 1.25f, 1.f);
            // render::fill_polygon2(polyloop::Polyloop2(poly.points), YELLOW, 0.f, 0.3f, false);
            render::draw_points(tensorField.getAllPoints(), ptData.color, 1.f, ptData.size);
            for (const field::Tensor<float> &t : tensorField.getAllTensors())
            {
                for (int i = 0; i < 4; ++i)
                    render::draw_vector(t.pos, t.dirs[i], vecData.color, vecData.scale, vecData.startThickness, vecData.endThickness, 0.f, vecData.color.a);
            }
            for (const auto &line : streamlines)
            {
                render::draw_bold_polyline2(line.points, lineData.color, 0.f, lineData.Thickness, lineData.color.a);
                // render::draw_points(line.points, ptData.color, ptData.color.a, ptData.size);
            }

            if (testTensor.dirs.size() == 4 && hasTestTensor)
            {
                for (int i = 0; i < 4; ++i)
                    render::draw_vector(testTensor.pos, testTensor.dirs[i], GREEN, vecData.scale, vecData.startThickness, vecData.endThickness, 0.f, vecData.color.a);
            }

            for(const auto &attr : attractors)
                attr.draw();
            

        },
        [&]() { // 二维屏幕空间绘图
            if (showIndices)
            {
                render.draw_index_fonts(tensorField.getAllPoints(), fontData.size, fontData.color);
                render.draw_index_fonts(tensorField.getCellCenters(), fontData.size, RED);
            }

            rlImGuiBegin(); // 开始ImGui帧渲染（必须在2D阶段调用）

            // 2. 自定义GUI窗口（纯2D固定在屏幕上）
            bool customOpen = true;
            if (ImGui::Begin("Render Settings", &customOpen))
            {
                if (ImGui::CollapsingHeader("Camera Control", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::SliderFloat("Camera Move Speed", &render::RENDER_MOVE_SPEED, 0.f, 2.f, "%.2f");
                    ImGui::SliderFloat("Camera Rotate Speed", &render::RENDER_ROTATE_SPEED, 0.f, 0.009f, "%.3f");
                    ImGui::SliderFloat("Camera Zoom Speed", &render::RENDER_ZOOM_SPEED, 0.f, 10.f, "%.2f");
                }

                if (ImGui::TreeNodeEx("Geo Settings", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::TreeNodeEx("Point Settings", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::SliderFloat("Point Size", &ptData.size, 0.01f, 10.f, "%.1f");
                        if (ImGui::ColorEdit4("Point Color", (float *)&ptData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                            ptData.syncFloatToColor();
                        ImGui::TreePop();
                    }
                    if (ImGui::TreeNodeEx("Vector Settings", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::SliderFloat("Vector Scale", &vecData.scale, 1.f, 20.f, "%.1f");
                        ImGui::SliderFloat("Start Thickness", &vecData.startThickness, 0.1f, 10.f, "%.1f");
                        ImGui::SliderFloat("End Thickness", &vecData.endThickness, 0.1f, 10.f, "%.1f");
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

                    ImGui::TreePop();
                }
            }
            ImGui::End();
            if (ImGui::Begin("TensorField", &customOpen))
            {
                if (ImGui::CollapsingHeader("PlotGenSetting", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Indent();
                    genPlot |= ImGui::SliderInt("PtNum", &ptNum, 4, 12);
                    genPlot |= ImGui::SliderFloat("Scale", &scale, 0.f, 2000.f, "%.1f");
                    genPlot |= ImGui::SliderFloat("Threshold", &threshold, 0.f, 1.f, "%.2f");
                    radiusChanged |= ImGui::SliderFloat("Attractor Radius", &radius, 1.f, 200.f, "%.1f");

                    ImGui::Unindent();
                }
                if (ImGui::CollapsingHeader("ShowSetting", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Indent();
                    ImGui::Checkbox("Show Indices", &showIndices);
                    ImGui::Unindent();
                }
            }
            ImGui::End();
            rlImGuiEnd();
        }});

    return 0;
}