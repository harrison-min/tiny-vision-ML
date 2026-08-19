#pragma once
#include "tensorMath.hpp"

namespace LossFunctionCalculator {
    template <typename T>
    Tensor<T> calculateGradient (const Tensor<T> & input, const Tensor<T> & expected);
};