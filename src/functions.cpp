#include <nn.hpp>

double NN::Sigmoid(double x) {
    double denominator = 1 + pow(Constants::euler, -x);
    return 1/denominator;
}

void NN::activation(Eigen::VectorXd &linearOutput) {
    for (auto &value : linearOutput) {
        value = NN::Sigmoid(value);
    }
}