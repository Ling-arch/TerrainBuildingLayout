
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include "chebyshevObject.h"
#include "util.h"
#include "geo.h"
#include "polyloop.h"
#include "render.h"
#include "renderUtil.h"



using namespace render;
using Vec2f = Eigen::Vector2f;
using Vec2i = Eigen::Vector2i;
using Vec = Eigen::VectorXf;
using namespace infinityVoronoi;


Eigen::Matrix2f createRotationMatrix2D(float angle_deg)
{
    // 步骤1：将角度制转换为弧度制（Eigen的三角函数默认用弧度）
    float angle_rad = angle_deg / 180.f * 3.1415926535897932384f;

    // 步骤2：计算旋转矩阵的元素
    float c = std::cos(angle_rad); // 余弦值
    float s = std::sin(angle_rad); // 正弦值

    // 步骤3：构造旋转矩阵（按逆时针公式赋值）
    Eigen::Matrix2f rot_mat;
    rot_mat << c, -s,
        s, c;

    return rot_mat;
}

int main()
{
    PointDrawData ptData;
    VectorDrawData vecData;
    FontDrawData fontData;
    LineDrawData lineData;
    rlSetClipPlanes(0.01, 100000.0);
    Renderer3D render(1920, 1080, 45.0f, CAMERA_PERSPECTIVE, "ChebyshevTest");
    render::RENDER_ZOOM_SPEED = 8.f;
    render::RENDER_MOVE_SPEED = 1.f;
    float extent = 10.f;

    static float initAxisAngle = 30.f;

    static Vec2f cutPlaneOrigin(0.5, 0);
    static Vec2f cutPlaneNormal(1, 0.6);
    bool reCut = false;/*  */
    bool reGenTriCut = false;
    static std::vector<float> scales = {2, 1, 1.5f, 0.5f};

    //----------------------------------
    // generate sites
    //----------------------------------
    Eigen::MatrixXf sites = chebyshevUtils::generateJitteredGridPoints(2, 2, extent);
    // std::cout << "sites compute finish" << std::endl;
    //----------------------------------
    // orientation function
    //----------------------------------
    infinityVoronoi::ChebyshevObject::OriFun oriFun =
        [](const Eigen::MatrixXf &xys)
    {
        int N = xys.rows();
        Eigen::VectorXf angles(N);

        for (int i = 0; i < N; ++i)
        {
            float x = xys(i, 0);
            float y = xys(i, 1);

            angles[i] = std::atan2(y, x);
        }

        return angles;
    };

    // std::cout << "oriFun compute finish" << std::endl;
    //----------------------------------
    // anisotropy function
    //----------------------------------
    infinityVoronoi::ChebyshevObject::ChebyshevObject::AniFun aniFun =
        [](const Eigen::MatrixXf &xys)
    {
        int N = xys.rows();

        Eigen::MatrixXf L = Eigen::MatrixXf::Ones(N, 4);

        // example: uniform weights
        for (int i = 0; i < N; ++i)
        {
            L(i, 0) = 2.f;
            L(i, 1) = 1.f;
            L(i, 2) = 2.f;
            L(i, 3) = 1.f;
        }

        return L;
    };

    // Vec2f cutLineDir = geo::rotate90CW(cutPlaneNormal);
    // cutLineDir.normalize();
    // Vec2f end = cutPlaneOrigin + cutLineDir * 1000;
    // Vec2f src = cutPlaneOrigin - cutLineDir * 1000;
    //----------------------------------
    // create object
    //----------------------------------
    MatrixXf testSites(2,2);
    testSites << 4.2,5.6,-7.1,-2.3;
    infinityVoronoi::ChebyshevObject2D obj(testSites, nullptr, nullptr, extent);
    //obj.debugPrintSitesAndLambdas();

    // std::vector<TriCutObject> triCuts;
    // for (int i = 0; i < 4; i++)
    //     triCuts.push_back({{0.f, 0.f}, i, scales, createRotationMatrix2D(initAxisAngle)});

    // for (auto &tri : triCuts)
    //     tri.cutWithPlane(cutPlaneOrigin, cutPlaneNormal, 4);

    // std::vector<polyloop::Polyloop2> cuttedPolys;
    // for (const auto &tri : triCuts)
    //     for (const auto &[k, _] : tri.polys)
    //         cuttedPolys.emplace_back(tri.buildPolyFromKey(k));

    //----------------------------------
    // compute diagram
    //----------------------------------/*  */
    //obj.computeDiagram();
    obj.computeNeighborsAndPlanes();
    obj.cutWithPlanes();
    // obj.lloydRelax(0.001f);
    obj.debugPrintSitesAndLambdas();
    rlImGuiSetup(true);
    render.runMainLoop(render::FrameCallbacks{
        [&]() { // 按键更新，重新绘图等事件，poly修改过需要重新fill
            // if (reGenTriCut || reCut)
            // {
            //     triCuts.clear();
            //     for (int i = 0; i < 4; i++)
            //         triCuts.push_back({{0.f, 0.f}, i, scales, createRotationMatrix2D(initAxisAngle)});

            //     for (auto &tri : triCuts)
            //         tri.cutWithPlane(cutPlaneOrigin, cutPlaneNormal, 4);
            //     cuttedPolys.clear();
            //     for (const auto &tri : triCuts)
            //         for (const auto &[k, _] : tri.polys)
            //             cuttedPolys.emplace_back(tri.buildPolyFromKey(k));

            //     cutLineDir = geo::rotate90CW(cutPlaneNormal);
            //     cutLineDir.normalize();
            //     end = cutPlaneOrigin + cutLineDir * 1000;
            //     src = cutPlaneOrigin - cutLineDir * 1000;
            // }
        },
        [&]() { // 3维空间绘图内容部分
            // for (int i = 0; i < cells.size(); i++)
            // {
            //     const auto &cell = cells[i];
            //     Color c = renderUtil::ColorFromHue(float(i) / cells.size());
            //     render::stroke_bold_polygon2(cell, c, 0.f, lineData.Thickness, c.a);
            //     render::fill_polygon2(cell, c, 0.f, 0.2f);
            //     render::draw_points(cell.points(), ptData.color, ptData.color.a, ptData.size);
            //     // render::stroke_bold_polygon2(cells[i], RL_BLACK, 0.f, 0.06f);
            // }
            obj.drawSiteWithDirs(RL_BLACK, lineData.Thickness);
            obj.drawCutPlanes(RL_RED,0.03f,1.5f);
            obj.drawCellSector(0,0);
            DrawGrid(30, 1.f);
            DrawLine3D({0, 0, 0}, {1000, 0, 0}, RL_RED);
            DrawLine3D({0, 0, 0}, {0, 0, -1000}, RL_GREEN);
            DrawLine3D({0, 0, 0}, {0, 1000, 0}, RL_BLUE);
            // DrawLine3D({src.x(), 0, -src.y()}, {end.x(), 0, -end.y()}, RL_DARKPURPLE);
            // for (int i = 0; i < cuttedPolys.size(); i++)
            // {
            //     const auto &cell = cuttedPolys[i];
            //     Color c = renderUtil::ColorFromHue(float(i) / cuttedPolys.size());
            //     render::stroke_bold_polygon2(cell, c, 0.f, lineData.Thickness, c.a);
            //     render::fill_polygon2(cell, c, 0.f, 0.2f);
            // }
        },
        [&]() { // 二维屏幕空间绘图
            // for (int i = 0; i < cells.size(); i++)
            // {
            //     const auto &cell = cells[i];

            //     render.draw_index_fonts(cell.points(), fontData.size, renderUtil::ColorFromHue(float(i) / cells.size()));
            // }
            rlImGuiBegin();
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
                        ImGui::SliderFloat("Point Size", &ptData.size, 0.01f, 10.f, "%.2f");
                        if (ImGui::ColorEdit4("Point Color", (float *)&ptData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                            ptData.syncFloatToColor();
                        ImGui::TreePop();
                    }
                    // if (ImGui::TreeNodeEx("Vector Settings", ImGuiTreeNodeFlags_DefaultOpen))
                    // {
                    //     ImGui::SliderFloat("Vector Scale", &vecData.scale, 1.f, 20.f, "%.1f");
                    //     ImGui::SliderFloat("Start Thickness", &vecData.startThickness, 0.1f, 10.f, "%.1f");
                    //     ImGui::SliderFloat("End Thickness", &vecData.endThickness, 0.1f, 10.f, "%.1f");
                    //     ImGui::SliderFloat("Vec Z", &vecZ, -100.0f, 100.f, "%.1f");
                    //     if (ImGui::ColorEdit4("Vector Color", (float *)&vecData.colorF, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
                    //         vecData.syncFloatToColor();
                    //     ImGui::TreePop();
                    // }

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
            if (ImGui::Begin("ChebyshevTest", &customOpen))
            {

                if (ImGui::TreeNodeEx("TriCutObjectTest", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::TreeNodeEx("CutPlane", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::Indent();
                        ImGui::Separator();
                        reCut |= ImGui::SliderFloat("OriginX", &cutPlaneOrigin.x(), 0.00f, 10.f, "%.2f");
                        reCut |= ImGui::SliderFloat("OriginY", &cutPlaneOrigin.y(), 0.00f, 10.f, "%.2f");
                        ImGui::Separator();
                        reCut |= ImGui::SliderFloat("NormalX", &cutPlaneNormal.x(), 0.00f, 10.f, "%.2f");
                        reCut |= ImGui::SliderFloat("NormalY", &cutPlaneNormal.y(), 0.00f, 10.f, "%.2f");
                        ImGui::Separator();

                        ImGui::Unindent();
                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNodeEx("TriCutObject", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::Indent();
                        ImGui::Separator();
                        reGenTriCut |= ImGui::SliderFloat("AxisAngle", &initAxisAngle, 0.0f, 180.f, "%.1f");
                        ImGui::Separator();
                        reGenTriCut |= ImGui::SliderFloat("PosX Scale", &scales[0], 0.00f, 10.f, "%.2f");
                        reGenTriCut |= ImGui::SliderFloat("PosY Scale", &scales[1], 0.00f, 10.f, "%.2f");
                        reGenTriCut |= ImGui::SliderFloat("NegX Scale", &scales[2], 0.00f, 10.f, "%.2f");
                        reGenTriCut |= ImGui::SliderFloat("NegY Scale", &scales[3], 0.00f, 10.f, "%.2f");
                        ImGui::Unindent();
                        ImGui::TreePop();
                    }
                    ImGui::TreePop();
                }
            }
            ImGui::End();
            rlImGuiEnd();
        }});

    return 0;
}