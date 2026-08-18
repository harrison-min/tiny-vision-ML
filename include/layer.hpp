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

template<typename T>
class ParameterLayer: public Layer<T> {
    public:
        virtual void updateBias (const Tensor<T> & newBias) = 0;
        virtual void updateWeights (const Tensor<T> & newWeights) = 0; 
        virtual const Tensor<T> & getWeights() const = 0;
        virtual const Tensor<T> & getBias() const = 0;
};

template <typename T>
class DenseLayer : public ParameterLayer<T>{
    private:
        Tensor<T> weights;
        Tensor<T> bias;
        Tensor<T> mostRecentInput;
    public:
        DenseLayer (int inputSize, int outputSize);
        Tensor<T> forward(const Tensor<T> & input) override;
        Tensor<T> backward(const Tensor<T> & gradient, T learningRate) override;

        void updateBias (const Tensor<T> & newBias) override;
        void updateWeights (const Tensor<T> & newWeights) override; 
        const Tensor<T> & getWeights() const override;
        const Tensor<T> & getBias() const override;
};

template <typename T>
class ConvolutionalLayer : public ParameterLayer <T> {
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

        void updateBias (const Tensor<T> & newBias) override;
        void updateWeights (const Tensor<T> & newWeights) override; 
        const Tensor<T> & getWeights() const override;
        const Tensor<T> & getBias() const override;
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
class FlattenLayer : public Layer <T> {
    private:
        std::vector<int> mostRecentDimensions;
    public:
        Tensor<T> forward (const Tensor<T> & input) override;
        Tensor<T> backward(const Tensor<T> & gradient, T learningRate) override;
};

template <typename T>
class MinMaxNormalizationLayer : public Layer <T> {
    private:
        int minRange;
        int maxRange;
        T minValue;
        T maxValue;
    public:
        MinMaxNormalizationLayer(int min = 0, int max = 1);
        Tensor<T> forward (const Tensor<T> & input) override;
        Tensor<T> backward(const Tensor<T> & gradient, T learningRate) override;
};