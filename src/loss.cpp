#include <loss.hpp>
#include <cmath>
#include <constants.hpp>


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
