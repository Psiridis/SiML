#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP
#include "loss_function.hpp"
#include "template_instantiator.hpp"
#include <memory>

namespace SiML
{
    class Model
    {
        public:
            virtual ~Model() = default;
    };

    template<Arithmetic T>
    class LinearRegression : public Model
    {
        public:
    };

#define X(T) EXTERN_INSTANTIATE_LINEAR_REGRESSION_FOR_TYPE(T)
    ARITHMETIC_TYPES
#undef X
}

#endif