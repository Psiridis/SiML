#include "loss_function.hpp"
#include <gtest/gtest.h>

static constexpr double dbl_eps = 1e-14;

TEST(MSETest, ComputeBasic) {
    SiML::MSE<double> mse;

    std::vector<double> y_true = {3.0, 5.0, 7.0};
    std::vector<double> y_pred = {2.5, 4.5, 6.5};

    // Expected MSE:
    // ((3-2.5)^2 + (5-4.5)^2 + (7-6.5)^2) / 3
    // = (0.25 + 0.25 + 0.25) / 3 = 0.75 / 3 = 0.25
    double expected = 0.25;

    double actual = mse.compute(y_pred, y_true);

    EXPECT_NEAR(actual, expected, 1e-9);
}

TEST(MSETest, ComputeZeroError) {
    SiML::MSE<double> mse;

    std::vector<double> y_true = {1.0, 2.0, 3.0};
    std::vector<double> y_pred = {1.0, 2.0, 3.0};

    // Expected MSE = 0 since predictions match true values exactly
    double expected = 0.0;

    double actual = mse.compute(y_pred, y_true);

    EXPECT_DOUBLE_EQ(actual, expected);
}

TEST(MSE_Compute, Compute_MSE_not_same_size_expection_handling){
    SiML::MSE<double> mse;
    constexpr std::array<double, 6> y_true({41, 45, 49, 47,44, 0.});
    constexpr std::array<double, 5> y_pred({43.6, 44.4, 45.2, 46, 46.8});
    EXPECT_THROW([&]{mse.compute(y_true, y_pred);}(), std::invalid_argument);
}

TEST(MSETest, DerivativeWRTWeights_SimpleCase) {
    SiML::MSE<double> mse;

    // Input features
    std::vector<double> X = {1.0, 2.0, 3.0};

    // x: 1D input vector; y = 2x + 1 (perfect prediction)
    std::vector<double> y_true = {3.0, 5.0, 7.0};       // y = 2x + 1
    std::vector<double> y_pred = {2.5, 4.5, 6.5};       // predicted y

    std::vector<double> expected_dL_dw = {-2.0}; // Known from hand calculation, size = 1

    auto dL_dw = mse.dL_dw(X, y_pred, y_true);

    ASSERT_EQ(dL_dw.size(), expected_dL_dw.size()); //size = 1 (only one feature)
    for (size_t i = 0; i < dL_dw.size(); ++i) {
        EXPECT_NEAR(dL_dw[i], expected_dL_dw[i], 1e-6);
    }
}

TEST(MSETest, DerivativeWRTWeights_TwoFeaturesFlatSpan) {
    SiML::MSE<double> mse;

    // 2 samples, 2 features → flattened row-major: [x11, x12, x21, x22]
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};         // shape (2, 2)
    std::vector<double> y_true = {11.0, 25.0};           // expected outputs
    std::vector<double> y_pred = {10.0, 24.0};           // underpredicts by 1.0

    // Errors = [1, 1]
    // grad[0] = -(2/2) * (1*1 + 3*1) = -4.0
    // grad[1] = -(2/2) * (2*1 + 4*1) = -6.0

    auto grad = mse.dL_dw(x, y_pred, y_true);

    ASSERT_EQ(grad.size(), 2);
    EXPECT_NEAR(grad[0], -4.0, 1e-6);
    EXPECT_NEAR(grad[1], -6.0, 1e-6);
}

TEST(MSETest, DerivativeWRTBias_OneFeature) {
    SiML::MSE<double> mse;

    std::vector<double> y_true = {3.0, 5.0, 7.0};
    std::vector<double> y_pred = {2.5, 4.5, 6.5};

    // Error = +0.5 for each → sum = 1.5 → dL/db = 2 * 1.5 / 3 = 1.0
    double expected = 1.0;
    double actual = mse.dL_db(y_pred, y_true);

    EXPECT_NEAR(actual, expected, 1e-6);
}