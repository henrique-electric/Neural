#include <loss.hpp>
#include <cmath>


namespace Loss {
    double SquareLoss(double result, double expected) {
        return pow(result - expected, 2);
    }

    double CrossEntropy(Eigen::VectorXd &prediction, Eigen::VectorXd &trueLabel) {
        double entropy = 0;
        for (int i=0; i < prediction.size(); i++) {
            entropy -= trueLabel(i) * log(prediction(i));
        }
        
        return entropy;
    }
}
