#include <loss.hpp>
#include <cmath>
#include <constants.hpp>


/*
	This file stores some loss functions and their derivatives that are used in the backpropagation process of the neural network. The loss functions 
    implemented here include the square loss, mean square loss, and cross-entropy loss. Each loss function 
    has a corresponding derivative function that calculates the gradient of the loss with respect to the predicted output, which is needed 
    for updating the weights and biases during training. The SigmoidDerivative function computes the derivative of the sigmoid activation function, 
    which is commonly used in neural networks. The WeightedSumDerivated function is a placeholder for calculating the derivative 
    of a weighted sum, which can be useful in certain architectures or custom layers.
*/

using namespace Constants;

static double Sigmoid(double x) {
    return 1/(1 + pow(euler, -x));
}

namespace Loss {
    double SquareLoss(double result, double expected) {
        return pow(result - expected, 2);
    }


    double MeanSquareLoss(Eigen::VectorXd &input, Eigen::VectorXd &expected) {
        double sum = 0;

        // Assuming input and trulabels have the same size
        for (int i=0; i < input.size(); i++) {
            sum += pow((input(i) - expected(i)), 2);
        }

        sum *= 1/input.size();
        return sum;
    }

    double MeanSquareLossDerivative(Eigen::VectorXd &input, Eigen::VectorXd &expected) {
        double sum = 0;
        for (int i=0; i < input.size(); i++) {
            sum += 2 * (input(i) - expected(i));
        }

        return sum;
    }

    double SquareLossDerivative(double result, double expected) {
        return 2 * (result - expected);
    }

    double SigmoidDerivative(double x) {
        return Sigmoid(x) * (1 - Sigmoid(x));
    }

    double WeightedSumDerivated(double x) {
        return x;   // return x itself since d/dw W * a + b = W
    }

    double CrossEntropy(Eigen::VectorXd &prediction, Eigen::VectorXd &trueLabel) {
        double entropy = 0;
        for (int i=0; i < prediction.size(); i++) {
            entropy -= trueLabel(i) * log(prediction(i));
        }
        
        return entropy;
    }
}
