#include <gtest/gtest.h>

#include <cmath>

#include <nn.hpp>

TEST(NeuralNetwork, ForwardProducesProbabilityOutput) {
    NN network(2, 5, 4, 3);

    network.forward();
    const Eigen::VectorXd output = network.getOutputLayer();

    ASSERT_EQ(output.size(), 3);
    EXPECT_NEAR(output.sum(), 1.0, 1e-12);

    for (int i = 0; i < output.size(); i++) {
        EXPECT_TRUE(std::isfinite(output(i)));
        EXPECT_GT(output(i), 0.0);
        EXPECT_LT(output(i), 1.0);
    }
}
