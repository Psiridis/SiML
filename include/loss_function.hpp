#ifndef LOSS_FUNCTION
#define LOSS_FUNCTION

#include "template_instantiator.hpp"
#include <span>
#include <vector>

namespace SiML
{
    template<Arithmetic T>
    class LossFunction{
        public:
            virtual ~LossFunction() = default;
            [[nodiscard]] virtual T compute(std::span<const T> y_pred, std::span<const T> y_true) const = 0;
    };


    template<Arithmetic T>
    class DifferentiableLossFunction : public LossFunction<T>{
        public:
            virtual ~DifferentiableLossFunction () = default;
            [[nodiscard]] virtual std::vector<T> dL_dw(std::span<const T> x, std::span<const T> y_pred, std::span<const T> y_true) const = 0;
            [[nodiscard]] virtual T dL_db(std::span<const T> y_pred, std::span<const T> y_true) const = 0;
    };


    template<typename T>
    class MSE : public DifferentiableLossFunction<T>{
        public:
            /**
            * @brief Computes the Mean Squared Error (MSE) loss between predicted and true values.
            *
            * Given predicted values \f$ \hat{y}_i \f$ and true target values \f$ y_i \f$, this function computes:
            * \f[
            * L = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
            * \f]
            * where \f$ m \f$ is the number of samples.
            *
            * This loss function measures the average squared difference between predicted and actual values.
            *
            * @tparam T Numeric type (e.g., float, double).
            * @param y_pred A span of predicted values \f$ \hat{y}_i \f$, of size \f$ m \f$.
            * @param y_true A span of true target values \f$ y_i \f$, of size \f$ m \f$.
            * @return The scalar Mean Squared Error loss.
            *
            * @throws std::invalid_argument If the spans do not have equal size.
            */
            [[nodiscard]] T compute(std::span<const T> y_pred, std::span<const T> y_true) const override;

            /**
            * @brief Computes the derivative of the Mean Squared Error (MSE) loss with respect to the weight vector.
            *
            * Given a flat input feature matrix \f$ X \f$, predicted values \f$ \hat{y} \f$, and true target values \f$ y \f$,
            * this function computes the gradient:
            * \f[
            * \frac{dL}{d\mathbf{w}} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i) \cdot \mathbf{x}_i
            * \f]
            * where \f$ m \f$ is the number of samples and \f$ \mathbf{x}_i \f$ is the feature vector for the i-th sample.
            *
            * The input feature matrix is provided in row-major flat format:
            * \f$ x = [x_{11}, x_{12}, ..., x_{1n}, x_{21}, ..., x_{mn}] \f$ where each row corresponds to one sample.
            *
            * @tparam T Numeric type (e.g., float, double).
            * @param x A span representing the flat feature matrix \f$ X \f$, of size \f$ m \times n \f$ (row-major order).
            * @param y_pred A span of predicted values \f$ \hat{y}_i \f$, of size \f$ m \f$.
            * @param y_true A span of true target values \f$ y_i \f$, of size \f$ m \f$.
            * @return A vector of size \f$ n \f$ representing the gradient of the MSE loss with respect to each weight.
            *
            * @throws std::invalid_argument If the spans do not have consistent sizes.
            */
            [[nodiscard]] std::vector<T> dL_dw(std::span<const T> x, std::span<const T> y_pred, std::span<const T> y_true) const override;


            /**
            * @brief Computes the derivative of the Mean Squared Error (MSE) loss with respect to the bias term.
            *
            * Given predicted values \f$ \hat{y} \f$ and true values \f$ y \f$, this function computes the gradient:
            * \f[
            * \frac{dL}{db} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)
            * \f]
            * This derivative is independent of the input features, and depends only on the prediction errors.
            *
            * @tparam T Numeric type (e.g., float, double).
            * @param y_pred A span of predicted values \f$ \hat{y}_i \f$, same size as @p y_true.
            * @param y_true A span of true target values \f$ y_i \f$, same size as @p y_pred.
            * @return The scalar gradient of the MSE loss with respect to the bias term.
            *
            * @throws std::invalid_argument If the spans are not of equal size.
            */
            [[nodiscard]] T dL_db(std::span<const T> y_pred, std::span<const T> y_true) const override;
        private:
    };

#define X(T) EXTERN_INSTANTIATE_MSE_FOR_TYPE(T)
    ARITHMETIC_TYPES
#undef X
}

#endif