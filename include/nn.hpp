#pragma once
#include <Eigen/Eigen>
#include <cmath>
#include <numbers>
#include <iostream>
#include <random>

namespace Constants {
    const double euler = 2.718281828459045f;
    const double pi = 3.14159265358979f;
}

class NN
{
    struct Layer {
        double output;
        Eigen::MatrixXd weights;
    };
    
    
private:
    Eigen::VectorXd inputs;
    Eigen::VectorX<Layer> layers;
    
public:
    static double Sigmoid(double x);
    static void activation(Eigen::VectorXd &linearOutput);
    
    void forward();

    NN(int layers, int neuronsPerLayer);
};
