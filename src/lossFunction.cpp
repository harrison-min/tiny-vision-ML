#include "lossFunction.hpp"
#include <cassert>


template <typename T>
Tensor<T> LossFunctionCalculator::calculateGradient (const Tensor<T> & input, const Tensor<T> & expected) {
    assert(input.getDimension() == expected.getDimension());
    
    Tensor<T> gradient(input.getOrder(), input.getDimension());

    size_t size = input.getSize();
    T divisor = 2.0/(static_cast<T>(size));


    for (size_t i = 0; i < size; ++ i) {
        gradient[i] = divisor * (input[i] - expected[i]);
    }

    return gradient;
}

namespace LossFunctionCalculator {
    template Tensor<double> calculateGradient (const Tensor<double> & input, const Tensor<double> & expected); 
    template Tensor<float> calculateGradient (const Tensor<float> & input, const Tensor<float> & expected); 
}