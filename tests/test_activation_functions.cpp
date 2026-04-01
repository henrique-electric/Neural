#include <gtest/gtest.h>

#include <cmath>

#include <nn.hpp>

namespace {
constexpr double kTolerance = 1e-9;
}

TEST(ActivationFunctions, SigmoidAtZeroIsHalf) {
    EXPECT_NEAR(NN::Sigmoid(0.0), 0.5, kTolerance);
}

TEST(ActivationFunctions, SigmoidIsMonotonicAroundZero) {
    EXPECT_LT(NN::Sigmoid(-1.0), NN::Sigmoid(0.0));
    EXPECT_LT(NN::Sigmoid(0.0), NN::Sigmoid(1.0));
}

TEST(ActivationFunctions, ReLUHandlesNegativeZeroAndPositive) {
    EXPECT_DOUBLE_EQ(NN::reLU(-10.0), 0.0);
    EXPECT_DOUBLE_EQ(NN::reLU(0.0), 0.0);
    EXPECT_DOUBLE_EQ(NN::reLU(3.25), 3.25);
}

TEST(ActivationFunctions, SoftmaxProducesValidDistribution) {
    Eigen::VectorXd values(3);
    values << 1.0, 2.0, 3.0;

    NN::Softmax(values);

    EXPECT_NEAR(values.sum(), 1.0, 1e-12);
    EXPECT_GT(values(0), 0.0);
    EXPECT_GT(values(1), 0.0);
    EXPECT_GT(values(2), 0.0);
    EXPECT_LT(values(0), values(1));
    EXPECT_LT(values(1), values(2));
}
