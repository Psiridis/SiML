#include <gtest/gtest.h>
#include "model.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <filesystem>

using namespace SiML;

TEST(SiML, LinearRegressionThrowsOnNullOptimizer) {
    LinearRegression model;
    Eigen::MatrixXd X(2,1);
    X << 1,2;
    Eigen::VectorXd y(2);
    y << 3,5;

    EXPECT_THROW(model.train(X, y, nullptr), std::invalid_argument);
}

TEST(SiML, LinearRegressionThrowsOnSizeMismatch) {
    LinearRegression model;
    auto loss = std::make_shared<MSE>();
    auto optimizer = std::make_shared<GradientDescent>(loss, 0.1, 100);

    Eigen::MatrixXd X(3,1);
    X << 1,2,3;
    Eigen::VectorXd y(2); // wrong size
    y << 3,5;

    EXPECT_THROW(model.train(X, y, optimizer), std::invalid_argument);
}

TEST(SiML, LinearRegressionThrowsOnPredictMismatch) {
    LinearRegression model;
    auto loss = std::make_shared<MSE>();
    auto optimizer = std::make_shared<GradientDescent>(loss, 0.1, 100);

    Eigen::MatrixXd X(3,1);
    X << 1,2,3;
    Eigen::VectorXd y(3);
    y << 3,5,7;

    model.train(X, y, optimizer);

    Eigen::MatrixXd X_bad(3,2); // wrong number of features
    X_bad << 1,2,3,4,5,6;

    EXPECT_THROW(model.predict(X_bad), std::invalid_argument);
}

TEST(SiML, LinearRegressionFitsSimpleLinearData) {
    LinearRegression model;
    auto loss = std::make_shared<MSE>();
    auto optimizer = std::make_shared<GradientDescent>(loss, 0.1, 500);

    // y = 2x + 1
    Eigen::MatrixXd X(4,1);
    X << 1,2,3,4;
    Eigen::VectorXd y(4);
    y << 3,5,7,9;

    model.train(X, y, optimizer);

    Eigen::VectorXd y_pred = model.predict(X);

    for (int i=0; i<y.size(); ++i) {
        EXPECT_NEAR(y[i], y_pred[i], 1e-2);
    }
}

TEST(SiML, LinearRegressionFitsMultiFeatureData) {
    LinearRegression model;
    auto loss = std::make_shared<MSE>();
    auto optimizer = std::make_shared<GradientDescent>(loss, 0.05, 2000);

    // y = 3*x1 + 2*x2 + 1
    Eigen::MatrixXd X(5,2);
    X << 1,2,
         2,1,
         3,0,
         0,3,
         4,2;
    Eigen::VectorXd y(5);
    y << 8,9,10,7,17;

    model.train(X, y, optimizer);

    Eigen::VectorXd y_pred = model.predict(X);

    for (int i=0; i<y.size(); ++i) {
        EXPECT_NEAR(y[i], y_pred[i], 1e-2);
    }
}