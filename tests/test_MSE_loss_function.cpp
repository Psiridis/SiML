#include <gtest/gtest.h>
#include "loss_function.hpp"


TEST(SiML, MSECompute) {
    SiML::MSE mse;

    // Simple dataset: y = 2x + 1
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;
    Eigen::VectorXd y(5);
    y << 3, 5, 7, 9, 11;
    Eigen::VectorXd y_pred(5);
    y_pred << 3.5, 5.5, 7.5, 9.5, 11.5;

    double mse_val = mse.compute(y, y_pred);

    EXPECT_NEAR(mse_val, 0.25, 1e-2); // tolerance 0.01
}