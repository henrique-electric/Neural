#pragma once
#include <Eigen/Eigen>

namespace Loss {
    double SquareLoss(double result, double expected);
    double MeanSquareLoss(Eigen::VectorXd &input, Eigen::VectorXd &expected);
    double CrossEntropy(Eigen::VectorXd &prediction, Eigen::VectorXd &trueLabel);
    
    double SigmoidDerivative(double x);
    double WeightedSumDerivated(double x);
    double SquareLossDerivative(double result, double expected);
    double MeanSquareLossDerivative(Eigen::VectorXd &input, Eigen::VectorXd &expected);

    enum LossOptions {
        CROSS_ENTROPY,
        MEAN_SQUARE_LOSS,
        SQUARE_LOSS,
    };
}
