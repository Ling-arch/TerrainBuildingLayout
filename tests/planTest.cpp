#include "util.h"
#include <Eigen/Dense>

using Eigen::Matrix2d;
using Eigen::Vector2d;
using std::cout, std::endl;

int main()
{
    Vector2d p0(-1, -1);
    Vector2d p1(1, 1);
    Vector2d ls(-1, 1);
    Vector2d ld(1, -1);

    Vector2d r;
    Matrix2d drda, drdb;
    std::tie(r, drda, drdb) = util::dw_intersection(p0, p1, ls, ld);

    cout << "Intersection point: " << r.transpose() << endl;
    cout << "Derivative wrt a: \n" << drda << endl;
    cout << "Derivative wrt b: \n" << drdb << endl;

    return 0;
}