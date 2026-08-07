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
    public:
        DenseLayer (int inputSize, int outputSize);
        Tensor<T> forward(const Tensor<T> & input) override;
        Tensor<T> backward(const Tensor<T> & gradient, T learningRate) override;
};

template <typename T>
class ActivationLayer : public Layer<T>{
    private:
        std::function<T(T)>  activationFunction;
    public:
        ActivationLayer(const std::function <T(T)> & func);
        Tensor<T> forward (const Tensor<T> & input) override;
        Tensor<T> backward(const Tensor<T> & gradient, T learningRate) override;
};