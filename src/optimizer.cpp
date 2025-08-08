#include "optimizer.hpp"
#include <iostream>

namespace SiML
{
    GradientDescent::GradientDescent(const std::shared_ptr<DifferentiableLossFunction> &loss_fun,
        double learning_rate, double epochs):
    m_loss_fun(loss_fun),
    m_learning_rate(learning_rate),
    m_max_epochs(epochs)
    {}

    void GradientDescent::optimize(const Eigen::MatrixXd &X,
                               const Eigen::VectorXd &y,
                               Eigen::VectorXd &weights, double &bias) const
    {
        const int m = X.rows(); // number of samples

        for (size_t epoch = 0; epoch < m_max_epochs; ++epoch)
        {
            // Predictions: y_pred = X * w + b
            const Eigen::VectorXd predictions = X * weights + Eigen::VectorXd::Ones(m) * bias;

            const Eigen::VectorXd dL_dy = m_loss_fun->gradient(y, predictions);
            const Eigen::VectorXd dw    = X.transpose() * dL_dy; // ∂L/∂w
            const double db             = dL_dy.sum();           // ∂L/∂b

            // Update parameters
            weights -= m_learning_rate * dw;
            bias    -= m_learning_rate * db;
        }
    }
}