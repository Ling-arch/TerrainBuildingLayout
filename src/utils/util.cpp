#include "util.h"

namespace util{

    tuple<Vector2d, Matrix2d, Matrix2d> dw_intersection_against_bisector(
        const Vector2d &ls, // line start
        const Vector2d &ld, // line dir
        const Vector2d &p0, // site0
        const Vector2d &p1  // site1
    )
    {
        const double half = 0.5;

        // middle point
        Vector2d a = (p0 + p1) * half;
        Vector2d b = rotate90(p1 - p0);

        Vector2d r;
        Matrix2d drda, drdb;
        std::tie(r, drda, drdb) = dw_intersection(ls, ld, a, b);

        Matrix2d dbdp0;
        dbdp0 << 0, 1, -1, 0;

        Matrix2d dbdp1;
        dbdp1 << 0, -1, 1, 0;

        // chain rule
        Matrix2d drdp0 = drda * half + drdb * dbdp0;
        Matrix2d drdp1 = drda * half + drdb * dbdp1;

        return {r, drdp0, drdp1};
    }

    tuple<Vector2d, Matrix2d, Matrix2d> dw_intersection(
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

        // 返回交点r和关于qs和qd的导数矩阵
        return {r, dr_dqs, dr_dqd};
    }

    CircumcenterResult wdw_circumcenter(
        const Vector2d &p0,
        const Vector2d &p1,
        const Vector2d &p2)
    {
        // 1) 计算边长平方
        double a0 = util::norm_squared(p1 - p2);
        double a1 = util::norm_squared(p2 - p0);
        double a2 = util::norm_squared(p0 - p1);

        // 2) 计算权重 b0,b1,b2
        double b0 = a0 * (a1 + a2 - a0);
        double b1 = a1 * (a2 + a0 - a1);
        double b2 = a2 * (a0 + a1 - a2);

        double sum = b0 + b1 + b2;
        double sum_inv = 1.0 / sum;

        // 归一化权重
        double c0 = b0 * sum_inv;
        double c1 = b1 * sum_inv;
        double c2 = b2 * sum_inv;

        // 3) 圆心
        Vector2d cc =
            c0 * p0 +
            c1 * p1 +
            c2 * p2;

        // --------------------
        // 4) db0, db1, db2 计算
        // --------------------
        double two = 2.0;

        array<Vector2d, 3> db0 = {
            (p0 - p2 + p0 - p1) * (two * a0),
            (p1 - p0 + p2 - p1) * (two * a0) + (p1 - p2) * (two * (a1 + a2 - a0)),
            (p2 - p0 + p1 - p2) * (two * a0) + (p2 - p1) * (two * (a1 + a2 - a0)),
        };

        array<Vector2d, 3> db1 = {
            (p0 - p1 + p2 - p0) * (two * a1) + (p0 - p2) * (two * (a2 + a0 - a1)),
            (p1 - p0 + p1 - p2) * (two * a1),
            (p2 - p1 + p0 - p2) * (two * a1) + (p2 - p0) * (two * (a2 + a0 - a1)),
        };

        array<Vector2d, 3> db2 = {
            (p0 - p2 + p1 - p0) * (two * a2) + (p0 - p1) * (two * (a0 + a1 - a2)),
            (p1 - p2 + p0 - p1) * (two * a2) + (p1 - p0) * (two * (a0 + a1 - a2)),
            (p2 - p1 + p2 - p0) * (two * a2),
        };

        // --------------------
        // 5) d(sum_inv)/dpi
        // --------------------
        double tmp = -1.0 / (sum * sum);

        array<Vector2d, 3> dsum_inv = {
            (db0[0] + db1[0] + db2[0]) * tmp,
            (db0[1] + db1[1] + db2[1]) * tmp,
            (db0[2] + db1[2] + db2[2]) * tmp,
        };

        // --------------------
        // 6) 最终的 dcc[i] = ?cc/?pi
        // --------------------
        array<Matrix2d, 3> dcc;

        // i = 0
        dcc[0] =
            c0 * Matrix2d::Identity() +
            p0 * (db0[0] * sum_inv + dsum_inv[0] * b0).transpose() +
            p1 * (db1[0] * sum_inv + dsum_inv[0] * b1).transpose() +
            p2 * (db2[0] * sum_inv + dsum_inv[0] * b2).transpose();

        // i = 1
        dcc[1] =
            c1 * Matrix2d::Identity() +
            p0 * (db0[1] * sum_inv + dsum_inv[1] * b0).transpose() +
            p1 * (db1[1] * sum_inv + dsum_inv[1] * b1).transpose() +
            p2 * (db2[1] * sum_inv + dsum_inv[1] * b2).transpose();

        // i = 2
        dcc[2] =
            c2 * Matrix2d::Identity() +
            p0 * (db0[2] * sum_inv + dsum_inv[2] * b0).transpose() +
            p1 * (db1[2] * sum_inv + dsum_inv[2] * b1).transpose() +
            p2 * (db2[2] * sum_inv + dsum_inv[2] * b2).transpose();

        return {cc, dcc};
    }

}