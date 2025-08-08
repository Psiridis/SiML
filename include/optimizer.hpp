#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP
#include "Eigen/Dense"
#include <memory>
#include "loss_function.hpp"

namespace SiML
{

    /**
     * @brief Abstract base class for optimization algorithms.
     *
     * Optimizers are responsible for adjusting model parameters
     * (weights and bias) to minimize a given loss function.
     */
    class Optimizer
    {
        public:
            /**
             * @brief Virtual destructor.
             */
            virtual ~Optimizer() = default;

            /**
             * @brief Optimize model parameters.
             *
             * Updates the provided @p weights and @p bias to reduce the loss
             * over the dataset (\p X, \p y).
             *
             * @param X Feature matrix of size (n_samples x n_features).
             * @param y Target vector of size (n_samples).
             * @param weights Reference to the model's weight vector.
             * @param bias Reference to the model's bias term.
             */
            virtual void optimize(const Eigen::MatrixXd &X,
                                  const Eigen::VectorXd &y,
                                  Eigen::VectorXd &weights,
                                  double &bias) const = 0;
    };


    /**
     * @brief Gradient Descent optimizer.
     *
     * Implements the standard batch gradient descent algorithm using a
     * differentiable loss function. At each iteration, the weights and bias
     * are updated in the opposite direction of the gradient of the loss
     * with respect to the parameters.
     */
    class GradientDescent : public Optimizer
    {
        public:
            /**
             * @brief Construct a new GradientDescent optimizer.
             *
             * @param loss_fun Shared pointer to a differentiable loss function.
             * @param learning_rate Step size for each update.
             * @param epochs Number of iterations over the full dataset.
             */
            explicit GradientDescent(const std::shared_ptr<DifferentiableLossFunction> &loss_fun,
                                     double learning_rate,
                                     double epochs);

            /**
             * @brief Run the gradient descent optimization process.
             *
             * Computes the gradient of the loss with respect to the model
             * parameters and updates them iteratively.
             *
             * @param X Feature matrix of size (n_samples x n_features).
             * @param y Target vector of size (n_samples).
             * @param weights Reference to the model's weight vector.
             * @param bias Reference to the model's bias term.
             */
            void optimize(const Eigen::MatrixXd &X,
                          const Eigen::VectorXd &y,
                          Eigen::VectorXd &weights,
                          double &bias) const override;

        private:
            std::shared_ptr<DifferentiableLossFunction> m_loss_fun; /**< Loss function used for gradient computation. */
            double m_learning_rate;                                 /**< Step size for parameter updates. */
            size_t m_max_epochs;                                    /**< Number of training iterations. */
    };

}

#endif