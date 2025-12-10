#include <inits.hpp>
#include <cmath>
#include <iostream>

double WInit::xavierInit(long numInputs, long numOutputs) {
    double xavierRes = sqrt(6/(numInputs + numOutputs));
    RNG::RNG rng(-xavierRes, xavierRes);
    
    return rng.rand();
}
