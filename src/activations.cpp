#include <nn.hpp>

using namespace Constants;

namespace ActivationFunctions {
    double sigmoid(double x) {
        double denominator = 1 + pow(Constants::euler, -x);
        return 1 / denominator;
    }

    double relu(double x) {
        if (x <= 0)
            return 0;
        else
            return x;
    }

    double lrelu(double x) {

        // Constant used on Leaky ReLU, change as needed
        double alphaConstant = 0.001;

        if (x > 0)
            return x;
        else
           return alphaConstant * x;
    }

    double elu(double x) {

        // Constant used on Exponential Linear Unit, change as needed
        double alphaConstant = 1; 

        if (x > 0)
            return x;
        else
            return alphaConstant * (pow(euler, x) - 1);
            
    }

    double softPlus(double x) {
        return log(1 + pow(euler, x));
    }

    void softmax(Eigen::VectorXd& vec) {
        double sum = 0;
        for (auto& x : vec)
            sum += pow(Constants::euler, x);

        for (auto& x : vec) {
            x = pow(Constants::euler, x) / sum;
        }
    }
}


namespace ActivationFunctionDerivative {
    double sigmoid(double x) {
        return ActivationFunctions::sigmoid(x) * (1 - ActivationFunctions::sigmoid(x));
    }
}