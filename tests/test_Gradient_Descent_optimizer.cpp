#include <gtest/gtest.h>
#include "optimizer.hpp"
#include "loss_function.hpp"


TEST(GradientDescent, Main) {
    auto mse = std::make_shared<SiML::MSE>();
    SiML::GradientDescent gd(mse, 0.08, 500);

    // Simple dataset: y = 2x + 1
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;
    Eigen::VectorXd y(5);
    y << 3, 5, 7, 9, 11;

    Eigen::VectorXd weights = Eigen::VectorXd::Zero(1);
    double bias = 0.;

    gd.optimize(X, y, weights, bias);

    EXPECT_NEAR(weights[0], 2., 1e-2); // tolerance 0.01
    EXPECT_NEAR(bias, 1., 1e-2);
}