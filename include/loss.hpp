#pragma once
#include <Eigen/Eigen>

/*
	Header file for the loss functions and their derivatives used in the backpropagation process of the neural network. The loss functions implemented here include the square loss, mean 
    square loss, and cross-entropy loss. Each loss function has a corresponding derivative function that calculates the gradient of the loss with respect to 
    the predicted output, which is needed for updating the weights and biases during training. The SigmoidDerivative function computes the derivative of the sigmoid 
    activation function, which is commonly used in neural networks. The WeightedSumDerivated function is a placeholder for calculating the derivative of a weighted sum, which 
    can be useful in certain architectures or custom layers.
*/

namespace Loss {
    double SquareLoss(double result, double expected);
    double MeanSquareLoss(Eigen::VectorXd &input, Eigen::VectorXd &expected);
    double CrossEntropy(Eigen::VectorXd &prediction, Eigen::VectorXd &trueLabel);
    
    double SigmoidDerivative(double x);
    double WeightedSumDerivated(double x);
    double SquareLossDerivative(double result, double expected);
double MeanSquareLossDerivative(Eigen::VectorXd &input, Eigen::VectorXd &expected);
}
