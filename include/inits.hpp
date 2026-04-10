#pragma once
#include <rng.hpp>

/*
	Header file for the init functions for the weights and biases of the neural network. The init functions implemented here include Xavier
*/

namespace WInit {
    double xavierInit(long numInputs, long numOutputs);
    double HeUniformInit(long numInputs);
}
