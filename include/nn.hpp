#pragma once
#define COMPILE_UTILS

#include <rng.hpp>
#include <constants.hpp>
#include <inits.hpp>
#include <loss.hpp>

#include <Eigen/Eigen>
#include <cmath>
#include <numbers>
#include <iostream>


enum OutputActivation {
    SOFTMAX,
    SIGMOID_OUT,
};

enum HiddenActivation {
    SIGMOID,
    RELU,
};

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
    
    Loss::LossOptions loss; // which loss to use
    OutputActivation outputLayerActivation;
    HiddenActivation hiddenLayerActivation;

    void handleOutActivation(void); // use for output layer
    double handleActivation(double out); // Use for hidden layers
    
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
    inline int getLayersNumber(void) { return layers.size(); };


#ifdef COMPILE_UTILS
    void printLayerWeights(void);
    void printLayerOutputs(void);
    void printLayerInputs(void);
    
    void printOutputLayerResult(void);
    void printOuputLayerInput(void);
#endif

    NN(int layers, int neuronsPerLayer, int inputSize, int outputSize, Loss::LossOptions loss, OutputActivation outLayerActivation, HiddenActivation hiddenActivation);
};
