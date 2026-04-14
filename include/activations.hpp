#pragma once
#include <constants.hpp>
#include <Eigen/Eigen>

namespace ActivationFunctions {
	double sigmoid(double x);
	double relu(double x);
	double lrelu(double x);
	double elu(double x);
	double softPlus(double x);
	void   softmax(Eigen::VectorXd &vec);
}

namespace ActivationFunctionDerivative {
	double sigmoid(double x);
}