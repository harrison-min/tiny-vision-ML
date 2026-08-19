#include "lossFunction.hpp"


template <typename T>
Tensor<T> LossFunctionCalculator<T>::calculateGradient (const Tensor<T> & input, const Tensor<T> & expected) {
    assert(input.getDimension() == expected.getDimension());
    
    Tensor<T> gradient(input.getOrder(), input.getDimension());

    size_t size = input.getSize();
    T divisor = 2.0/(static_cast<T>(size));


    for (size_t i = 0; i < size; ++ i) {
        gradient[i] = divisor * (input[i] - expected[i]);
    }

    return gradient;
}