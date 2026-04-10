#include <inits.hpp>
#include <cmath>
#include <iostream>

double WInit::xavierInit(long numInputs, long numOutputs) {
    double xavierRes = sqrt(6.0/(numInputs + numOutputs));
    RNG::RNG rng(-xavierRes, xavierRes);
    
    return rng.rand();
}

double WInit::HeUniformInit(long numImputs) {
    double heRes = sqrt(6/numImputs);
    RNG::RNG rng(-heRes, heRes);
    return rng.rand();
}
