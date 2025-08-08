#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP
#include "Eigen/Dense"

namespace SiML
{

    /**
     * @brief Abstract base class for loss functions.
     *
     * A loss function measures the difference between predicted values
     * and the actual target values. Lower loss values indicate better
     * model performance.
     */
    class LossFunction
    {
        public:
            /**
             * @brief Virtual destructor.
             */
            virtual ~LossFunction() = default;

            /**
             * @brief Compute the loss value.
             *
             * @param y_true Ground truth target values (size: n_samples).
             * @param y_pred Model predictions (size: n_samples).
             * @return double The computed loss value.
             *
             * @throws std::invalid_argument if @p y_true and @p y_pred sizes differ.
             */
            virtual double compute(const Eigen::VectorXd &y_true,
                                   const Eigen::VectorXd &y_pred) const = 0;
    };


    /**
     * @brief Abstract base class for differentiable loss functions.
     *
     * In addition to computing the loss value, differentiable loss
     * functions provide the gradient of the loss with respect to the
     * predictions, which is required for gradient-based optimizers.
     */
    class DifferentiableLossFunction : public LossFunction
    {
        public:
            /**
             * @brief Virtual destructor.
             */
            virtual ~DifferentiableLossFunction() = default;

            /**
             * @brief Compute the gradient of the loss with respect to predictions.
             *
             * @param y_true Ground truth target values (size: n_samples).
             * @param y_pred Model predictions (size: n_samples).
             * @return Eigen::VectorXd Gradient vector (∂L/∂y_pred).
             *
             * @throws std::invalid_argument if @p y_true and @p y_pred sizes differ.
             */
            virtual Eigen::VectorXd gradient(const Eigen::VectorXd &y_true,
                                             const Eigen::VectorXd &y_pred) const = 0;
    };


    /**
     * @brief Mean Squared Error (MSE) loss function.
     *
     * MSE is defined as:
     * \f[
     *   \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_{\text{pred},i} - y_{\text{true},i})^2
     * \f]
     * It is differentiable, with gradient:
     * \f[
     *   \frac{\partial L}{\partial y_{\text{pred}}} = \frac{2}{n} (y_{\text{pred}} - y_{\text{true}})
     * \f]
     */
    class MSE : public DifferentiableLossFunction
    {
        public:
            /**
             * @brief Compute the MSE loss.
             *
             * @param y_true Ground truth target values.
             * @param y_pred Model predictions.
             * @return double The mean squared error.
             *
             * @throws std::invalid_argument if input vector sizes differ.
             */
            double compute(const Eigen::VectorXd &y_true,
                           const Eigen::VectorXd &y_pred) const override;

            /**
             * @brief Compute the gradient of the MSE loss.
             *
             * @param y_true Ground truth target values.
             * @param y_pred Model predictions.
             * @return Eigen::VectorXd Gradient vector (∂L/∂y_pred).
             */
            Eigen::VectorXd gradient(const Eigen::VectorXd &y_true,
                                     const Eigen::VectorXd &y_pred) const override;
    };

}

#endif