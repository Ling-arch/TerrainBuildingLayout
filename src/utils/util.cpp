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

    Vector2d circumcenter(const Vector2d &p0, const Vector2d &p1, const Vector2d &p2){
        double a0 = (p0 - p1).squaredNorm();
        double a1 = (p1 - p2).squaredNorm();
        double a2 = (p2 - p0).squaredNorm();
        double b0 = a0 * (a1 + a2 - a0);
        double b1 = a1 * (a2 + a0 - a1);
        double b2 = a2 * (a0 + a1 - a2);
        double sum = b0 + b1 + b2;
        double sum_inv = 1.0 / sum;
        double c0 = b0 * sum_inv;
        double c1 = b1 * sum_inv;
        double c2 = b2 * sum_inv;
        return c0 * p0 + c1 * p1 + c2 * p2;
    }

    tuple<Vector2d, Matrix2d, Matrix2d> dw_intersection(
        const Vector2d &ps, // point on ray 1
        const Vector2d &pd, // direction of ray 1
        const Vector2d &qs, // point on ray 2
        const Vector2d &qd  // direction of ray 2
    ){
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

    Vector2d line_intersection(const Vector2d &ls, const Vector2d &ld,const Vector2d &ps, const Vector2d &pd){
        Vector2d qn = util::rotate90(pd);
        Vector2d a = ps - ls;

        double denom = ld.dot(qn);
        double b = 1.0 / denom;
        double t = a.dot(qn) * b;
        return ls + ld * t;
    }

    CircumcenterResult wdw_circumcenter(
        const Vector2d &p0,
        const Vector2d &p1,
        const Vector2d &p2)
    {
        // 1) 计算边长平方
        double a0 = (p1 - p2).squaredNorm();
        double a1 = (p2 - p0).squaredNorm();
        double a2 = (p0 - p1).squaredNorm();

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

    double polygon_area(const vector<Vector2d> &poly)
    {
        if (poly.size() < 3)
            return 0.0;
        double s = 0.0;
        for (size_t i = 0; i < poly.size(); ++i)
        {
            size_t j = (i + 1) % poly.size();
            s += poly[i].x() * poly[j].y() - poly[j].x() * poly[i].y();
        }
        return 0.5 * std::abs(s);
    }

    // Winding number (robust-ish) for point-in-polygon
    double winding_number(const vector<Vector2d> &poly, const Vector2d &p)
    {
        int wn = 0;

        for (size_t i = 0; i < poly.size(); ++i)
        {
            Vector2d a = poly[i];
            Vector2d b = poly[(i + 1) % poly.size()];

            Vector2d ab = b - a;
            Vector2d ap = p - a;
            double isLeft = ab.x() * ap.y() - ab.y() * ap.x();
            if (a.y() <= p.y())
            {
                if (b.y() > p.y())
                {
                    if (isLeft > 0)
                        ++wn;
                }
            }
            else
            {
                if (b.y() <= p.y())
                {
                    if (isLeft < 0)
                        --wn;
                }
            }
        }

        return static_cast<double>(wn);
    }


    vector<Vector2d> to_vec2_array(const vector<double> &flat){
        assert(flat.size() % 2 == 0);
        vector<Vector2d> out;
        out.reserve(flat.size() / 2);
        for (size_t i = 0; i + 1 < flat.size(); i += 2)
            out.emplace_back(flat[i], flat[i + 1]);
        return out;
    }

    vector<double> flat_vec2(const vector<Vector2d> &vec2arr)
    {
        vector<double> out;
        out.reserve(vec2arr.size() * 2);
        for (const auto &p : vec2arr)
        {
            out.push_back(p.x());
            out.push_back(p.y());
        }
        return out;
    }
}