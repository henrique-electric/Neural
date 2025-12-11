#pragma once

#include <rng.hpp>
#include <constants.hpp>
#include <inits.hpp>
#include <loss.hpp>

#include <Eigen/Eigen>
#include <cmath>
#include <numbers>
#include <iostream>



class NN
{
    struct Neuron {
        double output;
        Eigen::VectorXd weights;
    };
    
    struct Layer {
        Eigen::VectorXd output;
        Eigen::VectorXd input;
        Eigen::VectorX<Neuron> neurons;
    };
    
private:
    Eigen::VectorXd inputs;
    Eigen::VectorX<Layer> layers;
    Layer output;

    
    void fillMatrix(Eigen::MatrixXd &matrix, int rows, int columns);
    
public:
    static double Sigmoid(double x);
    static double reLU(double x);
    static void Softmax(Eigen::VectorXd &vec);
    
    static void activation(Eigen::VectorXd &linearOutput);
    
    void forward();

    NN(int layers, int neuronsPerLayer, int inputSize, int outputSize);
};
