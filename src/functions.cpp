#include <nn.hpp>

double NN::Sigmoid(double x) {
    double denominator = 1 + pow(Constants::euler, -x);
    std::cout << 1/denominator << std::endl;
    return 1/denominator;
}

double NN::reLU(double x) {
    if (x <= 0)
        return 0;
    else
        return x;
}

void NN::ComputeLayer(Layer &layer, Eigen::VectorXd input) {
    
}

void NN::activation(Eigen::VectorXd &linearOutput) {
    for (auto &value : linearOutput) {
        value = NN::Sigmoid(value);
    }
}
