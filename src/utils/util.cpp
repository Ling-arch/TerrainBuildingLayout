#include "util.h"

std::tuple<Vector2d, Matrix2d, Matrix2d> util::dw_intersection_against_bisector(
    const Vector2d &ls, // line start
    const Vector2d &ld, // line dir
    const Vector2d &p0, // site0
    const Vector2d &p1  // site1
)
{
    const double half = 0.5;

    // middle point
    Vector2d a = (p0 + p1) * half;
    Vector2d b = util::rotate90(p1-p0);

    Vector2d r;
    Matrix2d drda, drdb;
    std::tie(r, drda, drdb) = util::dw_intersection(ls, ld, a, b);

    Matrix2d dbdp0;
    dbdp0 << 0, 1, -1, 0;

    Matrix2d dbdp1;
    dbdp1 << 0, -1, 1, 0;

    // chain rule
    Matrix2d drdp0 = drda * half + drdb * dbdp0;
    Matrix2d drdp1 = drda * half + drdb * dbdp1;

    return {r, drdp0, drdp1};
}

std::tuple<Vector2d, Matrix2d, Matrix2d> util::dw_intersection(
    const Vector2d &ps, // point on ray 1
    const Vector2d &pd, // direction of ray 1
    const Vector2d &qs, // point on ray 2
    const Vector2d &qd  // direction of ray 2
)
{
    Vector2d qn = util::rotate90(qd);
    Vector2d a = qs - ps;

    double denom = pd.dot(qn);
    double b = 1.0 / denom;
    double t = a.dot(qn) * b;
    Vector2d r = ps + pd * t;
    Vector2d dt_dqn = a * b - pd * (a.dot(qn) * b * b);
    Vector2d dt_dqd = util::rotate90(dt_dqn);
    Vector2d dt_dqs = qn * b;

    Matrix2d dr_dqs = pd * dt_dqs.transpose();
    Matrix2d dr_dqd = pd * dt_dqd.transpose();

    //返回交点r和关于qs和qd的导数矩阵
    return {r, dr_dqs, dr_dqd};
}