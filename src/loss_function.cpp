#include "loss_function.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>

namespace SiML
{
    template<typename T>
    T MSE<T>::compute(std::span<const T> y_pred, std::span<const T> y_true) const
    {
        if (y_pred.size() != y_true.size()) {
            throw std::invalid_argument(std::string(__func__) + ": spans must be the same size.");
        }

        T cost = static_cast<T>(0.);

        const size_t m = y_pred.size();
        T sum = 0;
        for (size_t i=0; i<m; ++i) {
            T diff  = y_true[i] - y_pred[i];
            T diff2 = diff * diff;
            sum += diff2;
        }
        
        cost = sum / static_cast<T>(m);

        return cost;
    }


    template<typename T>
    std::vector<T> MSE<T>::dL_dw(std::span<const T> x, std::span<const T> y_pred, std::span<const T> y_true) const
    {
        if (y_pred.size() != y_true.size()) {
            throw std::invalid_argument(std::string(__func__) + ": spans must be the same size.");
        }

        const size_t m = y_true.size();
        const size_t n = x.size() / m;

        if (n*m != x.size()) {
            throw std::invalid_argument(std::string(__func__) + ": x, is not of the correct size.");
        }

        std::vector<T> grad(n, static_cast<T>(0));

        for (size_t i = 0; i < m; ++i) {
            T error = y_true[i] - y_pred[i];
            for (size_t j = 0; j < n; ++j) {
                grad[j] -= 2 * error * x[i*n+j];
            }
        }

        for (T &g : grad) {
            g /= static_cast<T>(m);
        }

        return grad;
    }


    template<typename T>
    T MSE<T>::dL_db(std::span<const T> y_pred, std::span<const T> y_true) const
    {
        if (y_pred.size() != y_true.size()) {
            throw std::invalid_argument(std::string(__func__) + ": spans must be the same size.");
        }

        T der = static_cast<T>(0.);

        const size_t m = y_pred.size();

        //Element-wise transformation: (y_true[i] - y_pred[i])
        T sum = std::transform_reduce(
            y_true.begin(), y_true.end(),
            y_pred.begin(),
            static_cast<T>(0),
            std::plus<>{},
            [](T y_t, T y_p) { return y_t - y_p; }
        ); 
        
        der = (static_cast<T>(2) / static_cast<T>(m)) * sum;

        return der;
    }

#define X(T) INSTANTIATE_MSE_FOR_TYPE(T)
    ARITHMETIC_TYPES
#undef X
} // namespace SiML
