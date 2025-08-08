#ifndef MODEL_HPP
#define MODEL_HPP
#include "Eigen/Dense"
#include <memory>
#include "optimizer.hpp"

namespace SiML
{
    /**
     * @brief Abstract base class for machine learning models.
     *
     * This class defines the interface that all models in the SiML
     * framework must implement. A model can be trained using a given
     * optimizer and can make predictions on new data.
     */
    class Model
    {
        public:
            /**
             * @brief Virtual destructor.
             */
            virtual ~Model() = default;

            /**
             * @brief Train the model on the provided dataset.
             *
             * @param X Feature matrix of size (n_samples x n_features).
             * @param y Target vector of size (n_samples).
             * @param optimizer Shared pointer to an Optimizer object used
             *        to adjust model parameters during training.
             *
             * @throws std::invalid_argument if @p optimizer is null or if
             *         the dimensions of @p X and @p y are inconsistent.
             */
            virtual void train(const Eigen::MatrixXd &X,
                               const Eigen::VectorXd &y,
                               const std::shared_ptr<Optimizer> &optimizer) = 0;

            /**
             * @brief Predict outputs for the given input data.
             *
             * @param X Feature matrix of size (n_samples x n_features).
             * @return Eigen::VectorXd Vector of predictions of size (n_samples).
             *
             * @throws std::invalid_argument if the number of features in @p X
             *         does not match the model's learned weights.
             */
            virtual Eigen::VectorXd predict(const Eigen::MatrixXd &X) const = 0;
    };


    /**
     * @brief Linear regression model.
     *
     * This class implements ordinary least squares linear regression
     * using the provided optimizer for parameter updates.
     */
    class LinearRegression : public Model
    {
        public:
            /**
             * @brief Train the linear regression model.
             *
             * Initializes weights and bias to zero (if not already set) and
             * then optimizes them using the provided optimizer.
             *
             * @param X Feature matrix of size (n_samples x n_features).
             * @param y Target vector of size (n_samples).
             * @param optimizer Shared pointer to an Optimizer object.
             */
            void train(const Eigen::MatrixXd &X,
                       const Eigen::VectorXd &y,
                       const std::shared_ptr<Optimizer> &optimizer) override;

            /**
             * @brief Predict target values for the given input features.
             *
             * Uses the learned weights and bias to compute predictions:
             * \f$ \hat{y} = X \cdot w + b \f$.
             *
             * @param X Feature matrix of size (n_samples x n_features).
             * @return Eigen::VectorXd Vector of predictions.
             */
            Eigen::VectorXd predict(const Eigen::MatrixXd &X) const override;

        private:
            Eigen::VectorXd m_weights; /**< Weight vector for each feature. */
            double m_bias;             /**< Bias term (intercept). */
    };
}

#endif