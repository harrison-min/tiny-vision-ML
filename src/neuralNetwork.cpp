#include "neuralNetwork.hpp"

template <typename T>
void NeuralNetwork<T>::addLayer(std::unique_ptr<Layer<T>> newLayer) {
    layers.push_back(std::move(newLayer));
}

template <typename T>
void NeuralNetwork<T>::backPropagate (const Tensor<T> & gradient, T learningRate) {
    Tensor<T> newGradient = gradient;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; -- i) {
        newGradient = layers[i]->backward(newGradient, learningRate);
    }
}

template <typename T>
void NeuralNetwork<T>::loadWeights (const std::vector<Tensor<T>> & newWeights) {
// TO DO	
}

template <typename T>
void NeuralNetwork<T>::loadBiases (const std::vector<Tensor<T>> & newBiases) {
// TO DO	
}

template <typename T>
Tensor<T> NeuralNetwork<T>::forward(const Tensor<T>& input) {
    Tensor<T> newInput = input;

    for (const auto& layer: layers) {
        newInput = layer->forward(newInput);
    }

    return newInput;
}
