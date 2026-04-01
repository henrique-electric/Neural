#include <gtest/gtest.h>

#include <cmath>

#include <inits.hpp>

TEST(WeightInitialization, XavierInitRespectsExpectedInterval) {
    const long numInputs = 8;
    const long numOutputs = 4;
    const double bound = std::sqrt(6.0 / static_cast<double>(numInputs + numOutputs));

    for (int i = 0; i < 256; i++) {
        const double value = WInit::xavierInit(numInputs, numOutputs);
        EXPECT_GE(value, -bound);
        EXPECT_LE(value, bound);
    }
}
