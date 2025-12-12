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
        double bias;
        Eigen::VectorXd weights;
    };
    
    struct Layer {
        Eigen::VectorXd output;
        Eigen::VectorXd input;
        Eigen::MatrixXd weights;
    };
    
private:
    Eigen::VectorXd inputs;
    Eigen::VectorX<Layer> layers;
    Layer output;
    
    Eigen::VectorXd trueLabels;
    

    
public:
    static double Sigmoid(double x);
    static double reLU(double x);
    static void Softmax(Eigen::VectorXd &vec);
    static void activation(Eigen::VectorXd &linearOutput);
    
    void GradientCalc(void);
    void forward();
    
    inline Eigen::VectorXd getOutputLayer(void) { return output.output; };
    inline void setInput(Eigen::VectorXd input) { inputs = input; };
    inline void setTrueLabels(Eigen::VectorXd labels) { trueLabels = labels; };

    NN(int layers, int neuronsPerLayer, int inputSize, int outputSize);
};
