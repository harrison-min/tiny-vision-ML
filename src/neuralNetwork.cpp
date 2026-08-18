#include "neuralNetwork.hpp"
#include <cassert>

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
    int weightsIndex = 0;
    for (size_t i = 0; i < layers.size(); ++ i) {
        auto layer = dynamic_cast<ParameterLayer<T>*>(layers[i].get());
        if (layer != nullptr) {
            assert(weightsIndex < static_cast<int>(newWeights.size()));
            layer->updateWeights(newWeights[weightsIndex++]);
        }
    }
    assert(weightsIndex == static_cast<int>(newWeights.size()));
}

template <typename T>
void NeuralNetwork<T>::loadBiases (const std::vector<Tensor<T>> & newBiases) {
    int biasIndex = 0;
    for (size_t i = 0; i < layers.size(); ++ i) {
        auto layer = dynamic_cast<ParameterLayer<T>*>(layers[i].get());
        if (layer != nullptr) {
            assert(biasIndex < static_cast<int>(newBiases.size()));
            layer->updateBias(newBiases[biasIndex++]);
        }
    }
    assert(biasIndex == static_cast<int>(newBiases.size()));
}

template <typename T>
Tensor<T> NeuralNetwork<T>::forward(const Tensor<T>& input) {
    Tensor<T> newInput = input;

    for (const auto& layer: layers) {
        newInput = layer->forward(newInput);
    }

    return newInput;
}


template <typename T>
std::vector<Tensor<T>> NeuralNetwork<T>::getWeights() {
    std::vector<Tensor<T>> weights;

    for (size_t i = 0; i < layers.size(); ++ i) {
        auto layer = dynamic_cast<ParameterLayer<T>*>(layers[i].get());
        if (layer != nullptr) {
            weights.push_back(layer->getWeights());
        }
    }

    return weights;
}

template <typename T>
std::vector<Tensor<T>> NeuralNetwork<T>::getBiases() {

    std::vector<Tensor<T>> biases;

    for (size_t i = 0; i < layers.size(); ++ i) {
        auto layer = dynamic_cast<ParameterLayer<T>*>(layers[i].get());
        if (layer != nullptr) {
            biases.push_back(layer->getBias());
        }
    }

    return biases;
}

template class NeuralNetwork<double>;
template class NeuralNetwork<float>;