#include <gtest/gtest.h>
#include "loss_function.hpp"

using namespace SiML;

// Test exact match → MSE should be 0
TEST(SiML, MSEComputeZeroError) {
    MSE mse;
    Eigen::VectorXd y_true(3), y_pred(3);
    y_true << 1.0, 2.0, 3.0;
    y_pred << 1.0, 2.0, 3.0;

    double result = mse.compute(y_true, y_pred);
    EXPECT_NEAR(result, 0.0, 1e-12);
}

// Test small constant error → easy to calculate manually
TEST(SiML, MSEComputeConstantError) {
    MSE mse;
    Eigen::VectorXd y_true(5), y_pred(5);
    y_true << 3, 5, 7, 9, 11;
    y_pred << 3.5, 5.5, 7.5, 9.5, 11.5;  // always +0.5 off

    // Errors: all = +0.5 → squared = 0.25 each
    // Sum = 1.25 → divide by 2n=10 → expected = 0.125
    double result = mse.compute(y_true, y_pred);
    EXPECT_NEAR(result, 0.125, 1e-12);
}

// Test mixed positive and negative errors
TEST(SiML, MSEComputeMixedErrors) {
    MSE mse;
    Eigen::VectorXd y_true(4), y_pred(4);
    y_true << 1, 2, 3, 4;
    y_pred << 2, 1, 4, 3;  // errors: +1, -1, +1, -1

    // Squared errors: 1, 1, 1, 1 → sum = 4
    // Divide by 2n = 8 → expected = 0.5
    double result = mse.compute(y_true, y_pred);
    EXPECT_NEAR(result, 0.5, 1e-12);
}

// Test exception on mismatched sizes
TEST(SiML, MSEComputeSizeMismatchThrows) {
    MSE mse;
    Eigen::VectorXd y_true(3), y_pred(2);
    y_true << 1, 2, 3;
    y_pred << 1, 2;

    EXPECT_THROW(mse.compute(y_true, y_pred), std::invalid_argument);
}

// Gradient should be zero if predictions are perfect
TEST(SiML, MSEZeroErrorGradient) {
    MSE mse;
    Eigen::VectorXd y_true(3), y_pred(3);
    y_true << 1.0, 2.0, 3.0;
    y_pred << 1.0, 2.0, 3.0;

    Eigen::VectorXd grad = mse.gradient(y_true, y_pred);
    Eigen::VectorXd expected = Eigen::VectorXd::Zero(3);

    for (int i = 0; i < grad.size(); ++i) {
        EXPECT_NEAR(grad[i], expected[i], 1e-12);
    }
}

// Gradient with constant positive error
TEST(SiML, MSEConstantErrorGradient) {
    MSE mse;
    Eigen::VectorXd y_true(4), y_pred(4);
    y_true << 1, 2, 3, 4;
    y_pred << 2, 3, 4, 5;  // always +1 error

    // (y_pred - y_true) = [1,1,1,1], / n=4 => [0.25,0.25,0.25,0.25]
    Eigen::VectorXd grad = mse.gradient(y_true, y_pred);
    Eigen::VectorXd expected(4);
    expected << 0.25, 0.25, 0.25, 0.25;

    for (int i = 0; i < grad.size(); ++i) {
        EXPECT_NEAR(grad[i], expected[i], 1e-12);
    }
}

// Gradient with mixed errors
TEST(SiML, MSEMixedErrorsGradient) {
    MSE mse;
    Eigen::VectorXd y_true(4), y_pred(4);
    y_true << 1, 2, 3, 4;
    y_pred << 2, 1, 4, 3;  // errors: +1, -1, +1, -1

    // (y_pred - y_true) = [1, -1, 1, -1], / n=4 => [0.25, -0.25, 0.25, -0.25]
    Eigen::VectorXd grad = mse.gradient(y_true, y_pred);
    Eigen::VectorXd expected(4);
    expected << 0.25, -0.25, 0.25, -0.25;

    for (int i = 0; i < grad.size(); ++i) {
        EXPECT_NEAR(grad[i], expected[i], 1e-12);
    }
}

// Test mismatch in size (should throw, like compute)
TEST(SiML, MSESizeMismatchThrowsGradient) {
    MSE mse;
    Eigen::VectorXd y_true(3), y_pred(2);
    y_true << 1, 2, 3;
    y_pred << 1, 2;

    EXPECT_THROW(mse.gradient(y_true, y_pred), std::invalid_argument);
}