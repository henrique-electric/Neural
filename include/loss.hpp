#pragma once
#include <Eigen/Eigen>

namespace Loss {
    double SquareLoss(double result, double expected);
    double CrossEntropy(Eigen::VectorXd &prediction, Eigen::VectorXd &trueLabel);
}
