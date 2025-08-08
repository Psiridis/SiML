#include <gtest/gtest.h>
#include "model.hpp"

TEST(SiML, LinearRegressionPredict) {
    SiML::LinearRegression lr;
    auto mse = std::make_shared<SiML::MSE>();

    auto opt = std::make_shared<SiML::GradientDescent>(mse, 0.08, 200);

    // Simple dataset: y = 2x + 1
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;

    Eigen::VectorXd y(5);
    y << 3, 5, 7, 9, 11;

    lr.train(X, y, opt);

    Eigen::VectorXd y_pred = lr.predict(X);

    for (int i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y_pred(i), y(i), 1e-2); // tolerance 0.01
    }
}