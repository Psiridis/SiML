#include "loss_function.hpp"
#include <iostream>

namespace SiML 
{
    double MSE::compute(const Eigen::VectorXd &y_true,
                             const Eigen::VectorXd &y_pred) const
    {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("y_true and y_pred must have the same size");
        }

        Eigen::VectorXd diff = (y_pred - y_true);

        return diff.squaredNorm() / static_cast<double>(y_true.size());
    }

    Eigen::VectorXd MSE::gradient(const Eigen::VectorXd &y_true,
                             const Eigen::VectorXd &y_pred) const
    {
        return (2.0 / static_cast<double>(y_true.size()))* (y_pred - y_true); // ∂L/∂y_pred
    }
}