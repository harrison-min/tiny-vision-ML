#pragma once
#include "tensorMath.hpp"
#include <functional>
 
template<typename T>
class Layer {
    public:
        virtual ~Layer() = default;
        virtual Tensor<T> forward(const Tensor<T> & input) = 0;
        virtual Tensor<T> backward(const Tensor<T> & gradient, T learningRate) = 0;
};

template <typename T>
class DenseLayer : public Layer<T>{
    private:
        Tensor<T> weights;
        Tensor<T> bias;
        Tensor<T> mostRecentInput;
    public:
        DenseLayer (int inputSize, int outputSize);
        Tensor<T> forward(const Tensor<T> & input) override;
        Tensor<T> backward(const Tensor<T> & gradient, T learningRate) override;
        void updateBias (const Tensor<T> & newBias);
        void updateWeights (const Tensor<T> & newWeights); 

        Tensor<T> getWeights();
        Tensor<T> getBias();
};

template <typename T>
class ActivationLayer : public Layer<T>{
    private:
        std::function<T(T)> activationFunction;
        std::function<T(T)> derivativeOfActivationFunction;
        Tensor<T> mostRecentInput;
    public:
        ActivationLayer(int inputSize, const std::function <T(T)> & actFunc, const std::function <T(T)> & derFunc);
        Tensor<T> forward (const Tensor<T> & input) override;
        Tensor<T> backward(const Tensor<T> & gradient, T learningRate) override;
};

template <typename T>
class ConvolutionalLayer : public Layer <T> {
    private:
        Tensor<T> weights;
        Tensor<T> mostRecentInput;
        Tensor<T> bias;
        int height;
        int width;
        int inputChannels;
        int numFilters;
    public:
        ConvolutionalLayer (int filterHeight, int filterWidth, int inChannels, int numberOfFilters);
        Tensor<T> forward (const Tensor<T> & input) override;
        Tensor<T> backward(const Tensor<T> & gradient, T learningRate) override;
        void updateBias (const Tensor<T> & newBias);
        void updateWeights (const Tensor<T> & newWeights); 

        Tensor<T> getWeights();
        Tensor<T> getBias();
};