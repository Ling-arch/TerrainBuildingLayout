#include "polyloop.h"

namespace polyloop
{
    Polyloop2::Polyloop2(const std::vector<Vector2> &points)
        : points_(points)
    {
        triangles_ = M2::triangulate_poly(points_);
    }

    Polyloop3::Polyloop3(const std::vector<Vector3> &points)
        : points_(points)
    {

        // 1. 计算平面（origin 和 normal）
        if (!M2::compute_plane(points_, normal_, 1e-5f))
        {
            // std::cerr << "Points are not coplanar or too few.\n";
            return;
        }

        // 2. 计算平面基 u,v
        M2::make_plane_basis(normal_, u_, v_);

        // 3. 投影到二维平面
        projected_points_ = M2::project_to_2d(points_, u_, v_);

        // 4. 三角化二维投影多边形
        triangles_ = M2::triangulate_poly(projected_points_);

        // std::cout << "Polyloop3 triangulated with " << triangles_.size() << " triangles.\n";
    }

   

    

    

   
}