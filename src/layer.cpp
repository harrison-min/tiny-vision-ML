#include "layer.hpp"
#include <cassert>

template <typename T>
ActivationLayer<T>::ActivationLayer(const std::function<T(T)> & func){
    activationFunction = func;
}

template <typename T>
Tensor<T> ActivationLayer<T>::forward (const Tensor<T> & input) {
    return input.apply(activationFunction);
}

template <typename T>
Tensor<T> ActivationLayer<T>::backward(const Tensor<T> & gradient, T learningRate) {
    assert(1==2); // ensures this crashes until we actually implement
    return gradient;
}

template class Layer<double>;
template class Layer<float>;
template class ActivationLayer<double>;
template class ActivationLayer<float>;