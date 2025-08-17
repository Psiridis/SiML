#include <gtest/gtest.h>
#include "optimizer.hpp"
#include "loss_function.hpp"

using namespace SiML;

// Test: perfect line y = 2x + 1 should be learned
TEST(SiML, GradientDescentLearnsLinearRelation) {
    // Training data
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;
    Eigen::VectorXd y(5);
    y << 3, 5, 7, 9, 11; // exactly y = 2x + 1

    // Initial params
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(1);
    double bias = 0.0;

    // Optimizer setup
    auto loss = std::make_shared<SiML::MSE>();
    GradientDescent gd(loss, 0.08, 300);

    gd.optimize(X, y, weights, bias);

    // Learned parameters should be close to (2, 1)
    EXPECT_NEAR(weights[0], 2.0, 1e-2);
    EXPECT_NEAR(bias, 1.0, 1e-2);

    // Predictions should match true y within tolerance
    Eigen::VectorXd y_pred = X * weights + Eigen::VectorXd::Ones(X.rows()) * bias;
    for (int i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y_pred[i], y[i], 1e-2);
    }
}

// Test: optimizer should reduce the loss after training
TEST(SiML, GradientDescentLossDecreases) {
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;
    Eigen::VectorXd y(5);
    y << 3, 5, 7, 9, 11;

    Eigen::VectorXd weights = Eigen::VectorXd::Zero(1);
    double bias = 0.0;

    // Optimizer setup
    auto loss = std::make_shared<SiML::MSE>();
    GradientDescent gd(loss, 0.08, 50);

    // Loss before optimization
    Eigen::VectorXd y_pred_init = X * weights + Eigen::VectorXd::Ones(X.rows()) * bias;
    double loss_before = loss->compute(y, y_pred_init);

    // Run optimization
    gd.optimize(X, y, weights, bias);

    // Loss after optimization
    Eigen::VectorXd y_pred_final = X * weights + Eigen::VectorXd::Ones(X.rows()) * bias;
    double loss_after = loss->compute(y, y_pred_final);

    EXPECT_LT(loss_after, loss_before);
}

// Test: when learning rate = 0, parameters should not change
TEST(SiML, GradientDescentNoUpdateWhenLearningRateZero) {
    Eigen::MatrixXd X(3, 1);
    X << 1, 2, 3;
    Eigen::VectorXd y(3);
    y << 2, 4, 6;

    Eigen::VectorXd weights = Eigen::VectorXd::Zero(1);
    double bias = 0.0;

    // Optimizer setup
    auto loss = std::make_shared<SiML::MSE>();
    GradientDescent gd(loss, 0.0, 10);

    gd.optimize(X, y, weights, bias);

    EXPECT_EQ(weights[0], 0.0);
    EXPECT_EQ(bias, 0.0);
}

TEST(SiML, GradientDescentTwoFeaturesTenSamples) {
    // Dataset: y = 3*x1 + 2*x2 + 1
    Eigen::MatrixXd X(13, 2);
    X << 1, 2,   // 3*1 + 2*2 + 1 = 8
         2, 1,   // 9
         3, 0,   // 10
         0, 3,   // 7
         4, 1,   // 15
         5, 2,   // 20
         2, 4,   // 15
         6, 0,   // 19
         1, 5,   // 14
         4, 2,   // 3*4 + 2*2 + 1 = 17
         8, 6,   // 3*8 + 2*6 + 1 = 24 + 12 + 1 = 37
        10, 2,   // 3*10 + 2*2 + 1 = 30 + 4 + 1 = 35
         3, 3;   // 16

    Eigen::VectorXd y(13);
    y << 8, 9, 10, 7, 15, 20, 15, 19, 14, 17, 37, 35, 16;

    // Initial weights and bias
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(2);
    double bias = 0.0;

    // Optimizer setup
    auto loss = std::make_shared<SiML::MSE>();
    GradientDescent gd(loss, 0.05, 2000);

    gd.optimize(X, y, weights, bias);


    // Check learned parameters (should be close to 3, 2, 1)
    EXPECT_NEAR(weights[0], 3.0, 1e-2);
    EXPECT_NEAR(weights[1], 2.0, 1e-2);
    EXPECT_NEAR(bias, 1.0, 1e-2);

    // Check predictions against ground truth
    Eigen::VectorXd y_pred = X * weights + Eigen::VectorXd::Ones(X.rows()) * bias;
    for (int i = 0; i < y.size(); i++) {
        EXPECT_NEAR(y_pred[i], y[i], 1e-2);
    }

    // --- Check loss is small ---
    double mse = loss->compute(y, y_pred);
    EXPECT_LT(mse, 1e-2);
}