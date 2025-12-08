#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <torch/torch.h>

namespace Util {
    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point_2 = K::Point_2;
    using Segment_2 = K::Segment_2;
    using Triangle_2 = K::Triangle_2;

    using MatrixXd = Eigen::MatrixXd;
    void printMatrix();
    
}
