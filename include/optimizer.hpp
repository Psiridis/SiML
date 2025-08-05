#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP
#include <memory>
#include "model.hpp"

namespace SiML
{
    class Optimizer
    {
        public:
            virtual ~Optimizer() = default;
            virtual void optimize(std::shared_ptr<Model> const &model) = 0;
    };

    class GradientDescent : public Optimizer
    {
        public:
        private:
    };
}

#endif