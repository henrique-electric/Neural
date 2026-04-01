#include <gtest/gtest.h>

#include <cmath>

#include <loss.hpp>

namespace {
constexpr double kTolerance = 1e-12;
}

TEST(LossFunctions, SquareLossIsZeroWhenPredictionMatchesExpected) {
    EXPECT_DOUBLE_EQ(Loss::SquareLoss(2.0, 2.0), 0.0);
}

TEST(LossFunctions, SquareLossMatchesAnalyticalValue) {
    EXPECT_DOUBLE_EQ(Loss::SquareLoss(5.0, 2.0), 9.0);
}

TEST(LossFunctions, CrossEntropyMatchesSingleHotExpectation) {
    Eigen::VectorXd prediction(3);
    prediction << 0.7, 0.2, 0.1;

    Eigen::VectorXd truth(3);
    truth << 1.0, 0.0, 0.0;

    const double expected = -std::log(0.7);
    EXPECT_NEAR(Loss::CrossEntropy(prediction, truth), expected, kTolerance);
}
