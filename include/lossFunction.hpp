#pragma once
#include "tensorMath.hpp"

template <typename T>
class LossFunctionCalculator {
    public:
        Tensor<T> calculateGradient (const Tensor<T> & input, const Tensor<T> & expected);
};