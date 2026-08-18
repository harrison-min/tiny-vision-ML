#pragma once
#include "layer.hpp"
#include <memory>
#include <vector>


template <typename T>
class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<Layer<T>>> layers;

    public:
        void addLayer(std::unique_ptr<Layer<T>> newLayer);
        Tensor<T> forward(const Tensor<T> & input);
        void backPropagate (const Tensor<T> & gradient, T learningRate);
        void loadWeights (const std::vector<Tensor<T>> & newWeights);
        void loadBiases (const std::vector<Tensor<T>> & newBiases);

        std::vector<Tensor<T>> getWeights();
        std::vector<Tensor<T>> getBiases();
};

