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
    int weightsIndex = 0;
    for (size_t i = 0; i < layers.size(); ++ i) {
        auto layer = dynamic_cast<ParameterLayer<T>*>(layers[i].get());
        if (layer != nullptr) {
            assert(weightsIndex < newWeights.size());
            layer->updateWeights(newWeights[weightsIndex++]);
        }
    }
    assert(weightsIndex == newWeights.size());
}

template <typename T>
void NeuralNetwork<T>::loadBiases (const std::vector<Tensor<T>> & newBiases) {
    int biasIndex = 0;
    for (size_t i = 0; i < layers.size(); ++ i) {
        auto layer = dynamic_cast<ParameterLayer<T>*>(layers[i].get());
        if (layer != nullptr) {
            assert(biasIndex < newBiases.size());
            layer->updateBias(newBiases[biasIndex++]);
        }
    }
    assert(biasIndex == newBiases.size());
}

template <typename T>
Tensor<T> NeuralNetwork<T>::forward(const Tensor<T>& input) {
    Tensor<T> newInput = input;

    for (const auto& layer: layers) {
        newInput = layer->forward(newInput);
    }

    return newInput;
}
