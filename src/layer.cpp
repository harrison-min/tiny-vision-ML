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

template <typename T>
ConvolutionalLayer<T>::ConvolutionalLayer (int filterHeight, int filterWidth, int inChannels, int numberOfFilters):
    weights(4, {numberOfFilters, inChannels, filterHeight, filterWidth}),
    mostRecentInput(4, {1, inChannels, filterHeight, filterWidth}),
    bias(1, {numberOfFilters}),
    height(filterHeight), 
    width(filterWidth), 
    inputChannels(inChannels), 
    numFilters(numberOfFilters) {
        std::random_device rd;
        std::mt19937 gen(rd());

        int inputSize = inputChannels * height * width;
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
Tensor<T> ConvolutionalLayer<T>::forward(const Tensor<T> & input) {
    mostRecentInput = input;

    std::vector<int> inputDimension = input.getDimension();
    
    int batchSize = inputDimension [0];
    assert(inputChannels == inputDimension[1]);
    int inputHeight = inputDimension [2];
    int inputWidth = inputDimension [3];

    int outputHeight = inputHeight - height + 1;
    int outputWidth = inputWidth - width + 1;

    Tensor<T> output (4, {batchSize, numFilters, outputHeight, outputWidth});

    for (int batch = 0; batch<batchSize; ++ batch) {
        for (int filter = 0; filter < numFilters; ++ filter){
            for (int yCoord = 0; yCoord < outputHeight; ++ yCoord) {
                for (int xCoord = 0; xCoord < outputWidth; ++ xCoord) {
                    T sum = bias[filter];
                    for (int channel = 0; channel < inputChannels; ++ channel) {
                        for (int i = 0; i < height; ++ i) {
                            for (int j = 0; j < width; ++ j) {
                                int inputIndex = batch * inputChannels * inputWidth * inputHeight +
                                                channel * inputHeight * inputWidth +
                                                (yCoord + i) * inputWidth +
                                                (xCoord + j);

                                int weightIndex = filter * inputChannels * height * width +
                                                channel * height * width +
                                                i * width +
                                                j;

                                sum += input[inputIndex] * weights[weightIndex];
                            }
                        }
                    }

                    int outputIndex = batch *numFilters * outputWidth * outputHeight +
                                    filter * outputHeight * outputWidth +
                                    yCoord * outputWidth +
                                    xCoord;
                    output[outputIndex] = sum;
                }
            }
        }
    }


    return output;
}

template <typename T>
Tensor<T> ConvolutionalLayer<T>::backward(const Tensor<T> & gradient, T learningRate) {
    std::vector<int> inputDimension = mostRecentInput.getDimension();

    int batchSize = inputDimension [0];
    assert(inputChannels == inputDimension[1]);
    int inputHeight = inputDimension [2];
    int inputWidth = inputDimension [3];

    int outputHeight = inputHeight - height + 1;
    int outputWidth = inputWidth - width + 1;

    Tensor<T> dBias (1, {numFilters});
    for (size_t i = 0; i < dBias.getSize(); ++ i) {
        dBias[i] = 0;
    }

    Tensor<T> dWeights (4, {numFilters, inputChannels, height, width});
    for (size_t i = 0; i < dWeights.getSize(); ++ i){
        dWeights[i] = 0;
    }

    Tensor<T> newGradient (4, {batchSize, inputChannels,inputHeight, inputWidth});
    for (size_t i = 0; i < newGradient.getSize(); ++ i){
        newGradient[i] = 0;
    }

    for (int batch = 0; batch < batchSize; ++ batch) {
        for (int filter = 0; filter < numFilters; ++ filter) {
            for (int yCoord = 0; yCoord < outputHeight; ++ yCoord) {
                for (int xCoord = 0; xCoord < outputWidth; ++ xCoord) {
                    int gradientIndex = batch * numFilters * outputHeight * outputWidth +
                                        filter * outputHeight * outputWidth +
                                        yCoord * outputWidth +
                                        xCoord;
                    dBias [filter] += gradient[gradientIndex];

                    for (int channel = 0; channel < inputChannels; ++ channel) {
                        for (int i = 0; i < height; ++ i) {
                            for (int j = 0; j < width; ++ j) {
                                int inputIndex = batch * inputChannels * inputHeight * inputWidth +
                                                channel * inputHeight * inputWidth +
                                                (yCoord + i) * inputWidth +
                                                xCoord + j;
                                
                                int weightIndex = filter * inputChannels * height * width +
                                                channel * height * width +
                                                i * width +
                                                j;

                                dWeights[weightIndex] += mostRecentInput[inputIndex] * gradient[gradientIndex];

                                newGradient[inputIndex] += weights[weightIndex] * gradient[gradientIndex];
                            }
                        }
                    }
                }
            }
        }
    }

    bias = bias - (dBias * learningRate);
    weights = weights - (dWeights * learningRate);

    return newGradient;
}

template <typename T>
void ConvolutionalLayer<T>::updateBias (const Tensor<T> & newBias) {
    assert(bias.getDimension() == newBias.getDimension());
    bias = newBias;
}

template <typename T>
void ConvolutionalLayer<T>::updateWeights (const Tensor<T> & newWeights) {
    assert(weights.getDimension() == newWeights.getDimension());
    weights = newWeights;
}

template <typename T>
Tensor<T> ConvolutionalLayer<T>::getWeights() {
    return weights;
}

template <typename T>
Tensor<T> ConvolutionalLayer<T>::getBias() {
    return bias;
}

template <typename T>
Tensor<T> FlattenLayer<T>::forward (const Tensor<T> & input) {
    mostRecentDimensions = input.getDimension();
    int batchSize = mostRecentDimensions[0];
    int vectorSize = 1;
    for (size_t i = 1; i < mostRecentDimensions.size(); ++ i)  {
        vectorSize *= mostRecentDimensions[i];
    }

    std::vector<int> outputDimensions = {batchSize, vectorSize};

    Tensor<T> output (2, outputDimensions);
    for (size_t i = 0; i < output.getSize(); ++ i) {
        output[i] = input[i];
    }

    return output;
}

template <typename T>
Tensor<T> FlattenLayer<T>::backward(const Tensor<T> & gradient, T learningRate) {
    std::vector<int> gradientDimensions = gradient.getDimension();
    int gradientBatchSize = gradientDimensions[0];
    int gradientVectorSize = gradientDimensions[1];

    int inputBatchSize = mostRecentDimensions[0];
    int inputVectorSize = 1;
    for (size_t i = 1; i < mostRecentDimensions.size(); ++ i)  {
        inputVectorSize *= mostRecentDimensions[i];
    }

    assert(gradientBatchSize == inputBatchSize && gradientVectorSize == inputVectorSize); 

    Tensor<T> newGradient (mostRecentDimensions.size(), mostRecentDimensions);
    for (size_t i = 0; i < newGradient.getSize(); ++ i) {
        newGradient[i] = gradient[i];
    }

    return newGradient;
}

template class FlattenLayer<double>;
template class FlattenLayer<float>;
template class ConvolutionalLayer<double>;
template class ConvolutionalLayer<float>;
template class ActivationLayer<double>;
template class ActivationLayer<float>;
template class DenseLayer<double>;
template class DenseLayer<float>;