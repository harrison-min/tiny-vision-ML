#include "layer.hpp"
#include <cassert>
#include <random>
#include <cmath>

template <typename T>
ActivationLayer<T>::ActivationLayer(int inputSize, const std::function <T(T)> & actFunc, const std::function <T(T)> & derFunc):
    mostRecentInput(2,{1,inputSize}){
    activationFunction = actFunc;
    derivativeOfActivationFunction = derFunc;
    for (size_t i = 0; i < mostRecentInput.getSize(); ++ i) {
        mostRecentInput[i] = 0;
    }
}

template <typename T>
Tensor<T> ActivationLayer<T>::forward (const Tensor<T> & input) {
    mostRecentInput = input;
    return input.apply(activationFunction);
}

template <typename T>
Tensor<T> ActivationLayer<T>::backward(const Tensor<T> & gradient, T learningRate) {
    Tensor<T> derivative = mostRecentInput.apply(derivativeOfActivationFunction);
    return TensorCalculator::hadamardProduct(gradient,derivative);
}


template<typename T>
DenseLayer<T>::DenseLayer(int inputSize, int outputSize):
    weights(2, {inputSize, outputSize}),
    bias(1, {outputSize}),
    mostRecentInput (2, {1, inputSize}) {
    
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
        
        for (size_t i = 0; i < mostRecentInput.getSize(); ++ i) {
            mostRecentInput[i] = 0;
        }
}

template <typename T>
void DenseLayer<T>::updateBias (const Tensor<T> & newBias) {
    assert(newBias.getDimension() == bias.getDimension());
    bias = newBias;
}

template <typename T>
void DenseLayer<T>::updateWeights (const Tensor<T> & newWeights) {
    assert(newWeights.getDimension() == weights.getDimension());
    weights = newWeights;
}

template <typename T>
Tensor<T> DenseLayer<T>::forward(const Tensor<T> & input) {
    mostRecentInput = input;
    return (input * weights) + bias;
}

template <typename T>
Tensor<T> DenseLayer<T>::backward(const Tensor<T> & gradient, T learningRate) {
    Tensor<T> dWeights = mostRecentInput.transpose() * gradient; 
    Tensor<T> dBias = gradient.collapse({0}, TensorCalculator::sum<T>, 0.0); 
    Tensor<T> newGradient = gradient * weights.transpose();

    weights = weights - (dWeights * learningRate);
    bias = bias - (dBias * learningRate);
    return newGradient;
}

template <typename T>
Tensor<T> DenseLayer<T>::getWeights() {
    return weights;
}

template <typename T>
Tensor<T> DenseLayer<T>::getBias() {
    return bias;
}

template class ActivationLayer<double>;
template class ActivationLayer<float>;
template class DenseLayer<double>;
template class DenseLayer<float>;