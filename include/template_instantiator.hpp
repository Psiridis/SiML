#ifndef TEMPLATE_INSTANTIATOR_HPP
#define TEMPLATE_INSTANTIATOR_HPP

#include <type_traits>

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

#define ARITHMETIC_TYPES \
    X(short)              \
    X(int)                \
    X(long)               \
    X(long long)          \
    X(unsigned short)     \
    X(unsigned int)       \
    X(unsigned long)      \
    X(unsigned long long) \
    X(float)              \
    X(double)             \
    X(long double)

#define EXTERN_INSTANTIATE_MSE_FOR_TYPE(T) extern template class MSE<T>;
#define INSTANTIATE_MSE_FOR_TYPE(T)        template class MSE<T>;

#define EXTERN_INSTANTIATE_LINEAR_REGRESSION_FOR_TYPE(T) extern template class LinearRegression<T>;
#define INSTANTIATE_LINEAR_REGRESSION_FOR_TYPE(T)        template class LinearRegression<T>;

#endif // TEMPLATE_INSTANTIATOR_HPP