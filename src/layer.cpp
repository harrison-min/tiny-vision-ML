#include "layer.hpp"
#include <cassert>
#include <random>
#include <cmath>

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
    assert(false); // ensures this crashes until we actually implement
    return gradient;
}


template<typename T>
DenseLayer<T>::DenseLayer(int inputSize, int outputSize):
    weights(2, {inputSize, outputSize}),
    bias(1, {outputSize}) {
    
        std::random_device rd;
        std::mt19937 gen(rd());

        const T stdDev = static_cast<T>(std::sqrt(2.0/inputSize));
        std::normal_distribution<T> dist(0.0, stdDev);

        for (size_t i = 0; i < weights.getSize(); ++ i) {
            weights[i] = dist(gen);
        }

        for (size_t i = 0; i < bias.getSize(); ++ i) {
            bias[i] = 0;
        }
}

template <typename T>
Tensor<T> DenseLayer<T>::forward(const Tensor<T> & input) {
    return (input * weights) + bias;
}

template <typename T>
Tensor<T> DenseLayer<T>::backward(const Tensor<T> & gradient, T learningRate) {
    assert(false); // ensures this crashes until we actually implement
    return gradient;
}

template class ActivationLayer<double>;
template class ActivationLayer<float>;
template class DenseLayer<double>;
template class DenseLayer<float>;