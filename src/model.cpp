#include "model.hpp"
#include <iostream>

namespace SiML
{
    void LinearRegression::train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                                const std::shared_ptr<Optimizer> &optimizer) 
    {
        if (!optimizer) {
            throw std::invalid_argument("Optimizer cannot be null");
        }

        if (X.rows() != y.size()) {
            throw std::invalid_argument("Number of rows in X must match the size of y");
        }

        // Initialize parameters if not already set
        if (m_weights.size() != X.cols()) {
            m_weights = Eigen::VectorXd::Zero(X.cols());
        }
        m_bias = 0.0;

        // Let the optimizer adjust the parameters
        optimizer->optimize(X, y, m_weights, m_bias);
    }

    Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &X) const
    {
        if (X.cols() != m_weights.size()) {
            throw std::invalid_argument("Number of features in X does not match model weights");
        }

        return (X * m_weights).array() + m_bias;
    }
}