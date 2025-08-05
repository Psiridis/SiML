#include "model.hpp"

namespace SiML
{
#define X(T) INSTANTIATE_LINEAR_REGRESSION_FOR_TYPE(T)
    ARITHMETIC_TYPES
#undef X
} // namespace SiML