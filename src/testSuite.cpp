#include "testSuite.hpp"
#include "layer.hpp"
#include "neuralNetwork.hpp"
#include "imageEncoder.hpp"
#include "lossFunction.hpp"
#include <iostream>
#include <limits>
#include <typeinfo>


//==================================================================
//  TENSOR TESTS
//==================================================================

static const int TOTAL_NUMBER_OF_TENSOR_TESTS = 8;
template class TensorTestSuite<double>;
template class TensorTestSuite<float>;

template <typename T>
TensorTestSuite<T>::TensorTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_TENSOR_TESTS;
}

template <typename T>
void TensorTestSuite<T>::run() {
    std::cout << "\n\nRUNNING TENSOR TEST SUITE:\n";
    std::cout << "Type: " << typeid(T).name() << "\n";
    if (innerProductTest()) numPassed ++;
    if (transposeTest()) numPassed ++;
    if (matrixAdditionTest()) numPassed ++;
    if (matrixMultiplicationTest()) numPassed ++;
    if (hadamardProductTest()) numPassed ++;
    if (biasAdditionTest()) numPassed ++;
    if (ReLUTest()) numPassed ++;
    if (sumCollapseTest()) numPassed ++;

    assert(numPassed == numTests);
}

template <typename T>
Tensor<T> TensorTestSuite<T>::generateTestTensor(const std::vector<int> & dimension) {
    Tensor<T> testTensor(dimension.size(), dimension);
    int totalSize = testTensor.getSize();

    for (int i = 0; i < totalSize; ++ i) {
        testTensor[i] = static_cast<T>(i - totalSize/2);
    }

    return testTensor;
}

template <typename T>
bool TensorTestSuite<T>::innerProductTest(){
    std::vector<int> dimension = {3, 3, 3, 3};
    Tensor<T> tensor = generateTestTensor(dimension);
    Tensor<T> zeroTensor = generateTestTensor(dimension);

    double expectedDouble = 0;
    
    const int size = static_cast<int>(tensor.getSize());
    for (int i = 0; i < size; ++ i) {
        zeroTensor[i] = 0;
        expectedDouble += tensor[i] * tensor[i];
    }

    bool testPassed = true;

    static const T epsilon = std::numeric_limits<T>::epsilon() * 10;

    if (std::abs(TensorCalculator::innerProduct(tensor, tensor) - expectedDouble) > epsilon) {
        std::cout << "FAIL: innerProductTest (Inner Product of type "<< typeid(T).name() << " Tensor with itself doesnt match expected)\n";
        testPassed = false;
    }

    if (std::abs(TensorCalculator::innerProduct(tensor, zeroTensor)) > epsilon) {
        std::cout << "FAIL: innerProductTest (Inner Product of "<< typeid(T).name() << " Tensor with 0 tensor isnt 0)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: innerProductTest\n";
    }

    return testPassed;
}

template <typename T>
bool TensorTestSuite<T>::transposeTest(){
    std::vector<int> dimension = {3, 2, 2, 4};
    Tensor<T> tensor = generateTestTensor(dimension);
 
	bool testPassed = true;

    if (tensor.transpose().transpose() != tensor) {
        std::cout << "FAIL: transposeTest (" << typeid(T).name()<<" Tensor Transpose of the Transpose doesnt equal the original)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: transposeTest\n";
    }

    return testPassed;
}

template <typename T>
bool TensorTestSuite<T>::matrixAdditionTest(){
    std::vector<int> dimension = {3, 2, 2, 4};
    Tensor<T> tensor = generateTestTensor(dimension);

    Tensor<T> expectedTensor = generateTestTensor(dimension);
    Tensor<T> zeroTensor = generateTestTensor(dimension);

    const int size = static_cast<int>(tensor.getSize());
    for (int i = 0; i < size; ++ i) {
        expectedTensor[i] = 2 * tensor[i];
        zeroTensor[i] = 0;
    }

	bool testPassed = true;
    if (tensor + tensor != expectedTensor) {
        std::cout << "FAIL: matrixAdditionTest ( " << typeid(T).name() << " tensor added to itself doenst match expected)\n";
        testPassed = false;
    }

    if (tensor + zeroTensor != tensor) {
        std::cout << "FAIL: matrixAdditionTest ( " << typeid(T).name() << " tensor added to 0 tensor doenst equal itself)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: matrixAdditionTest\n";
    }

    return testPassed;
}

template <typename T>
bool TensorTestSuite<T>::matrixMultiplicationTest(){
 
    std::vector<int> dimension = {3, 2, 4, 4};
    const int dimensionSize = static_cast<int>(dimension.size());
    std::vector<int> stride (dimensionSize);
    stride[dimensionSize - 1] = 1;

    for (int i = dimensionSize - 2; i >=0; --i) {
        stride[i] = dimension[i + 1] * stride[i + 1];
    }

    Tensor<T> tensor = generateTestTensor(dimension);

    Tensor<T> identityTensor = generateTestTensor(dimension);
    Tensor<T> zeroTensor = generateTestTensor(dimension);

    const int size = static_cast<int>(tensor.getSize());

    for (int i = 0; i < size; ++ i) {
        zeroTensor[i] = 0;

        int row = (i / stride[dimensionSize - 2]) %dimension[dimensionSize - 2];
        int col = (i / stride[dimensionSize - 1]) %dimension[dimensionSize - 1];

        bool isIdentity = (row == col);

        if (isIdentity) {
            identityTensor[i] = 1;
        } else {
            identityTensor[i] = 0;
        }
    }


	bool testPassed = true;

    if (tensor * identityTensor != tensor) {
        std::cout << "FAIL: matrixMultiplicationTest ( " << typeid(T).name() << " tensor multiplied to identity doenst match itself)\n";
        testPassed = false;
    }

    if (tensor * zeroTensor != zeroTensor) {
        std::cout << "FAIL: matrixMultiplicationTest ( " << typeid(T).name() << "  tensor multiplied to 0 Tensor doenst equal 0 tensor)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: matrixMultiplicationTest\n";
    }

    return testPassed;
}

template <typename T>
bool TensorTestSuite<T>::hadamardProductTest(){
    std::vector<int> dimension = {3, 2, 3, 4, 5};
    Tensor<T> tensor = generateTestTensor(dimension);

    Tensor<T> oneTensor = generateTestTensor(dimension);
    Tensor<T> zeroTensor = generateTestTensor(dimension);
    const int size = static_cast<int>(tensor.getSize());

    for (int i = 0; i < size; ++ i) {
        oneTensor[i] = 1;
        zeroTensor[i] = 0;
    }

	bool testPassed = true;

    if (TensorCalculator::hadamardProduct(tensor, oneTensor) != tensor) {
        std::cout << "FAIL: hadamardProductTest ( " << typeid(T).name() << " tensor multiplied to 1 tensor doenst match itself)\n";
        testPassed = false;
    }

    if (TensorCalculator::hadamardProduct(tensor, zeroTensor) != zeroTensor) {
        std::cout << "FAIL: hadamardProductTest ( " << typeid(T).name() << " tensor multiplied to 0 tensor doenst equal zero)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: hadamardProductTest\n";
    }

    return testPassed;
}

template <typename T>
bool TensorTestSuite<T>::biasAdditionTest(){
    const int columnLength = 5;
    std::vector<int> dimension = {3, 2, 3, 4, columnLength};
    Tensor<T> tensor = generateTestTensor(dimension);
    
    Tensor<T> expectedTensor = generateTestTensor(dimension);

    const int size = static_cast<int>(tensor.getSize());

    for (int i = 0; i < size; ++ i) {
        expectedTensor[i] = tensor[i] + 1;
    }

    std::vector<int> biasDimension = {columnLength};
    Tensor<T> oneBiasVector = generateTestTensor(biasDimension);
    Tensor<T> zeroBiasVector = generateTestTensor(biasDimension);

    for (int i = 0; i < columnLength; ++ i) {
        oneBiasVector[i] = 1; 
        zeroBiasVector[i] = 0;
    }

	bool testPassed = true;
    if (tensor + oneBiasVector != expectedTensor) {
        std::cout << "FAIL: biasAdditionTest ( " << typeid(T).name() << " tensor added with bias doenst match the expected)\n";
        testPassed = false;
    }

    if (tensor + zeroBiasVector != tensor) {
        std::cout << "FAIL: biasAdditionTest ( " << typeid(T).name() << " tensor added with 0 bias doenst match itself)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout<< "PASS: biasAdditionTest\n";
    }

    return testPassed;
}

template <typename T>
bool TensorTestSuite<T>::ReLUTest(){
    std::vector<int> dimension = {3, 2, 3, 4, 7};
    Tensor<T> tensor = generateTestTensor(dimension);
    
    Tensor<T> expectedTensor = generateTestTensor(dimension);

    const int size = static_cast<int>(tensor.getSize());

    for (int i = 0; i < size; ++ i) {
        if (tensor[i] < 0) {
            expectedTensor[i] = 0;
        } else {
            expectedTensor[i] = tensor[i];
        }
    }

	bool testPassed = true;
    if (tensor.apply(TensorCalculator::reLU<T>) != expectedTensor) {
        std::cout << "FAIL: ReLUTest (ReLU application on  " << typeid(T).name() << " Tensor did not match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: ReLUTest\n";
    }

    return testPassed;
}

template <typename T>
bool TensorTestSuite<T>::sumCollapseTest(){
    std::vector<int> dimension = {2, 3, 4, 3, 4};
    std::vector<int> eliminatedDimensions = {3, 4};

    std::vector<int> collapsedDimensions = dimension;
    for (int i = 0; i < static_cast<int>(eliminatedDimensions.size()); ++ i) {
        collapsedDimensions.erase(collapsedDimensions.begin() + eliminatedDimensions[i] - i);
    }

    Tensor<T> tensor = generateTestTensor(dimension);
    Tensor<T> expectedTensor = generateTestTensor(collapsedDimensions);

    const int expectedSize = static_cast<int>(expectedTensor.getSize());
    int innerSize = 1;
    for (auto dim : eliminatedDimensions) {
        innerSize *= dimension[dim];
    }

    for (int i = 0; i < expectedSize; ++ i) {
        T TSum = 0;
        for (int j = 0; j < innerSize; ++j) {
            TSum += tensor[i * innerSize + j];
        }
        expectedTensor[i] = TSum;
    }

	bool testPassed = true;
    if (tensor.collapse(eliminatedDimensions, TensorCalculator::sum<T>, 0) != expectedTensor) {
        std::cout << "FAIL: sumCollapseTest (Sum Collapse on  " << typeid(T).name() << " Tensor did not match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: sumCollapseTest\n";
    }
    return testPassed;
}


//==================================================================
//  LAYER TESTS
//==================================================================

static const int TOTAL_NUMBER_OF_LAYER_TESTS = 10;
template class LayerTestSuite<double>;
template class LayerTestSuite<float>;

template <typename T>
LayerTestSuite<T>::LayerTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_LAYER_TESTS;
}

template <typename T>
void LayerTestSuite<T>::run() {
    std::cout << "\n\nRUNNING LAYER TEST SUITE:\n";
    std::cout << "Type: " << typeid(T).name() << "\n";
    if (forwardDenseLayerTest()) numPassed ++;
    if (backwardDenseLayerTest()) numPassed ++;
    if (forwardActivationLayerTest()) numPassed ++;
    if (backwardActivationLayerTest()) numPassed ++;
    if (forwardConvolutionalLayerTest()) numPassed ++;
    if (backwardConvolutionalLayerTest()) numPassed ++;
    if (forwardFlattenLayerTest()) numPassed ++;
    if (backwardFlattenLayerTest()) numPassed ++;
    if (forwardMinMaxNormalizationLayerTest()) numPassed ++;
    if (backwardMinMaxNormalizationLayerTest()) numPassed ++;
    assert (numTests == numPassed);
}

template <typename T>
bool LayerTestSuite<T>::forwardDenseLayerTest() {
    const int inputSize = 3;
    const int outputSize = 2;
    const int batchSize = 5;
    DenseLayer<T> layer(inputSize, outputSize);
    Tensor<T> weights (2, {inputSize, outputSize});
    Tensor<T> bias (1, {outputSize});
    Tensor<T> input(2, {batchSize, inputSize});
    Tensor<T> expectedOutput(2, {batchSize,outputSize});

    for (size_t i = 0; i < weights.getSize(); ++ i) {
        weights[i] = -1.0;
    }

    for (size_t i = 0; i < bias.getSize(); ++ i) {
        bias [i] = 1.5;
    }

    for (size_t i = 0; i < expectedOutput.getSize(); ++ i) {
        expectedOutput[i] = -1.5;
    }

    for (size_t i = 0; i < input.getSize(); ++ i) {
        input [i] = 1.0;
    }
    layer.updateBias(bias);
    layer.updateWeights(weights);
    Tensor<T> output = layer.forward(input);

    bool testPassed = true;
    if (output != expectedOutput) {
        std::cout << "FAIL: forwardDenseLayerTest ( " << typeid(T).name() << " output tensor does not match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: forwardDenseLayerTest\n";
    }

    return testPassed;
}
   
template <typename T>
bool LayerTestSuite<T>::backwardDenseLayerTest() {
    const int inputSize = 3;
    const int outputSize = 2;
    const int batchSize = 5;
    const T learningRate = 1.0;

    DenseLayer<T> layer(inputSize, outputSize);

    Tensor<T> weights (2, {inputSize, outputSize});
    Tensor<T> bias (1, {outputSize});
    Tensor<T> input(2, {batchSize, inputSize});
    Tensor<T> gradient(2, {batchSize, outputSize});
    Tensor<T> expectedOutputGradient(2, {batchSize, inputSize});
    Tensor<T> expectedWeights(2, {inputSize, outputSize});
    Tensor<T> expectedBias (1, {outputSize});

    for (size_t i = 0; i < weights.getSize(); ++ i) {
        weights[i] = -1.0;
    }

    for (size_t i = 0; i < gradient.getSize(); ++ i) {
        gradient[i] = 1.0;
    }

    for (size_t i = 0; i < bias.getSize(); ++ i) {
        bias [i] = 1.5;
        T dB = 1.0 * batchSize;
        expectedBias[i] = bias[i] - dB * learningRate;
    }

    for (size_t i = 0; i < input.getSize(); ++ i) {
        input [i] = 1.0;
    }

    expectedOutputGradient = gradient * weights.transpose();

    Tensor<T> dW = input.transpose() * gradient;

    expectedWeights = weights - (dW * learningRate);

    layer.updateBias(bias);
    layer.updateWeights(weights);

    layer.forward(input);
    Tensor<T> resultGradient = layer.backward(gradient, learningRate);

    bool testPassed = true;
    if (resultGradient != expectedOutputGradient) {
        std::cout << "FAIL: backwardDenseLayerTest ( " << typeid(T).name() << " gradient doesnt match expected)\n";
        testPassed = false;
    }

    if (layer.getWeights()!= expectedWeights) {
        std::cout << "FAIL: backwardDenseLayerTest ( " << typeid(T).name() << " weights doesnt match expected)\n";
        testPassed = false;
    }

    if (layer.getBias()!= expectedBias) {
        std::cout << "FAIL: backwardDenseLayerTest ( " << typeid(T).name() << " bias doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: backwardDenseLayerTest\n";
    }

    return testPassed;
}

template <typename T>
bool LayerTestSuite<T>::forwardActivationLayerTest() {
    const int inputSize = 5;
    ActivationLayer<T> layer (inputSize, TensorCalculator::reLU<T>, TensorCalculator::derivativeReLU<T>);

    Tensor<T> input (2, {1, inputSize});
    Tensor<T> expected (2, {1, inputSize});

    const int size = static_cast<int>(input.getSize());
    for (int i = 0; i < size; ++ i) {
        input [i] = i - size/2;
        if (i - size/2 <= 0) {
            expected[i] = 0.0;
        } else {
            expected[i] = i - size/2;
        }
    }

    Tensor<T> result = layer.forward(input);

    bool testPassed = true;
    if (result != expected) {
        std::cout << "FAIL: forwardActivationLayerTest ( " << typeid(T).name() << " result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: forwardActivationLayerTest\n";
    }

    return testPassed;
}

template <typename T>
bool LayerTestSuite<T>::backwardActivationLayerTest() {
    const int inputSize = 5;
    ActivationLayer<T> layer (inputSize, TensorCalculator::reLU<T>, TensorCalculator::derivativeReLU<T>);

    Tensor<T> input (2, {1, inputSize});
    Tensor<T> gradient (2, {1, inputSize});
    Tensor<T> expected (2, {1, inputSize});

    const int size = static_cast<int>(input.getSize());
    for (int i = 0; i < size; ++ i) {
        input [i] = i - size/2;
        gradient[i] = i;
        T derivativeValue = 0.0;
        if (i - size/2 > 0) {
            derivativeValue = 1.0;
        } 
        expected[i]  = derivativeValue * gradient[i];
    }

    const T learningRate = 1.0; // these dont really apply to this layer
    layer.forward(input);
    Tensor<T> result = layer.backward(gradient, learningRate);

    bool testPassed = true;
    if (result != expected) {
        std::cout << "FAIL: backwardActivationLayerTest ( " << typeid(T).name() << " result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: backwardActivationLayerTest\n";
    }

    return testPassed;
}

template <typename T>
bool LayerTestSuite<T>::forwardConvolutionalLayerTest() {
    const int imageHeight = 4;
    const int imageWidth = 4;
    const int inputChannels = 3;
    const int numFilters = 2;
    const int filterSize = 3;
    const int batchSize = 1;

    std::vector<int> imageDimensions = {batchSize, inputChannels, imageHeight, imageWidth};
    Tensor<T> image (imageDimensions.size(), imageDimensions);
    int imageSize = static_cast<int>(image.getSize());

    for (int i = 0; i < imageSize; ++ i) {
        image [i] = static_cast<T>(i);
    }

    Tensor<T> bias (1, {numFilters});
    Tensor<T> weights (4, {numFilters, inputChannels, filterSize, filterSize});

    for (size_t i = 0; i < bias.getSize(); ++ i) {
        bias[i] = static_cast<T>(0.0);
    }

    for (size_t i = 0; i < weights.getSize(); ++ i) {
        weights[i] = static_cast<T>(1.0);
    }

    ConvolutionalLayer<T> layer(filterSize, filterSize, inputChannels, numFilters);
    layer.updateBias(bias);
    layer.updateWeights(weights);
    Tensor<T> result = layer.forward(image);

    std::vector<int> expectedDimensions = {batchSize, numFilters, 2, 2};
    Tensor<T> expected(expectedDimensions.size(), expectedDimensions);
    T expectedValues[] = {567, 594, 675, 702, 567, 594, 675, 702};//precomputed expected values
    for (size_t i = 0; i < expected.getSize(); ++ i) {
        expected [i] = expectedValues [i];
    }

    bool testPassed = true;
    if (result != expected) {
        std::cout << "FAIL: forwardConvolutionalLayerTest ( " << typeid(T).name() << " result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: forwardConvolutionalLayerTest\n";
    }

    return testPassed;
}

template <typename T>
bool LayerTestSuite<T>::backwardConvolutionalLayerTest() {
    const T learningRate = 1.0;
    const int imageHeight = 4;
    const int imageWidth = 4;
    const int inputChannels = 3;
    const int numFilters = 2;
    const int filterSize = 3;
    const int batchSize = 1;

    std::vector<int> imageDimensions = {batchSize, inputChannels, imageHeight, imageWidth};
    Tensor<T> image (imageDimensions.size(), imageDimensions);
    int imageSize = static_cast<int>(image.getSize());

    for (int i = 0; i < imageSize; ++ i) {
        image [i] = static_cast<T>(1.0);
    }

    Tensor<T> bias (1, {numFilters});
    Tensor<T> weights (4, {numFilters, inputChannels, filterSize, filterSize});

    for (size_t i = 0; i < bias.getSize(); ++ i) {
        bias[i] = static_cast<T>(0.0);
    }

    for (size_t i = 0; i < weights.getSize(); ++ i) {
        weights[i] = static_cast<T>(1.0);
    }

    std::vector<int> gradientDimensions = {batchSize, numFilters, 2, 2};
    Tensor<T> gradient(gradientDimensions.size(), gradientDimensions);

    for (size_t i = 0; i < gradient.getSize(); ++ i) {
        gradient[i] = static_cast<T>(1.0);
    }

    Tensor<T> expectedBias (1, {numFilters});
    for (size_t i = 0; i < expectedBias.getSize(); ++ i) {
        expectedBias[i] = static_cast<T>(-4.0); //starting bias is 0.0 - gradient: 1.0 * learnign rate * 1.0
    }

    Tensor<T> expectedWeights(4, {numFilters, inputChannels, filterSize, filterSize});
    for (size_t i = 0; i < expectedWeights.getSize(); ++ i)     {
        expectedWeights[i] = static_cast<T>(-3.0);
    }

    Tensor<T> expectedGradient (imageDimensions.size(), imageDimensions);
    T expectedGradientValues[] = {
        2, 4, 4, 2,
        4, 8, 8, 4,
        4, 8, 8, 4,
        2, 4, 4, 2};
    for (size_t i = 0; i < expectedGradient.getSize(); ++ i) {
        expectedGradient[i] = static_cast<T>(expectedGradientValues[i%(imageHeight * imageWidth)]);
    }

    ConvolutionalLayer<T> layer (filterSize, filterSize, inputChannels, numFilters);

    layer.updateWeights(weights);
    layer.updateBias(bias);
    layer.forward(image);
    Tensor<T> resultGradient = layer.backward (gradient, learningRate);

    bool testPassed = true;
    if (expectedGradient != resultGradient) {
        std::cout << "FAIL: backwardConvolutionalLayerTest ( " << typeid(T).name() << " gradient doesnt match expected)\n";
        testPassed = false;
    }

    if (expectedBias != layer.getBias()) {
        std::cout << "FAIL: backwardConvolutionalLayerTest ( " << typeid(T).name() << " bias doesnt match expected)\n";
        testPassed = false;
    }

    if (expectedWeights != layer.getWeights()) {
        std::cout << "FAIL: backwardConvolutionalLayerTest ( " << typeid(T).name() << " weights doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: backwardConvolutionalLayerTest\n";
    }

    return testPassed;
}


template <typename T>
bool LayerTestSuite<T>::forwardFlattenLayerTest() {
    std::vector<int> inputDimension = {2, 2, 3, 4};
    Tensor<T> input (inputDimension.size(), inputDimension);
    Tensor<T> expected (2, {2, 24});

    for (size_t i = 0; i < input.getSize(); ++ i) {
        input[i] = static_cast<T>(i);
        expected[i] = static_cast<T>(i);
    }

    FlattenLayer<T> layer;
    Tensor<T> result = layer.forward(input);
    
    bool testPassed = true;
    if (result != expected) {
        std::cout << "FAIL: forwardFlattenLayerTest ( " << typeid(T).name() << " result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: forwardFlattenLayerTest\n";
    }

    return testPassed;
}

template <typename T>
bool LayerTestSuite<T>::backwardFlattenLayerTest() {
    std::vector<int> inputDimension = {2, 2, 3, 4};
    Tensor<T> input (inputDimension.size(), inputDimension);
    std::vector<int> gradientDimension = {2, 24};
    Tensor<T> gradient (gradientDimension.size(), gradientDimension);
    Tensor<T> expected (4, {2, 2, 3, 4});

    for (size_t i = 0; i < input.getSize(); ++ i) {
        input[i] = static_cast<T>(i);
        expected[i] = static_cast<T>(i);
        gradient[i] = static_cast<T>(i);
    }

    FlattenLayer<T> layer;
    layer.forward(input);
    Tensor<T> result = layer.backward(gradient, 0.0);
    
    bool testPassed = true;
    if (result != expected) {
        std::cout << "FAIL: backwardFlattenLayerTest ( " << typeid(T).name() << " result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: backwardFlattenLayerTest\n";
    }

    return testPassed;
}

template <typename T>
bool LayerTestSuite<T>::forwardMinMaxNormalizationLayerTest() {
    std::vector <int> dimensions = {2, 2, 2};

    Tensor<T> input (dimensions.size(), dimensions);
    Tensor<T> expected (dimensions.size(), dimensions);
    T expectedValues[] = {
        0, 1.0/7.0,
        2.0/7.0, 3.0/7.0,
        4.0/7.0, 5.0/7.0,
        6.0/7.0, 1.0
    };

    for (size_t i = 0; i < input.getSize(); ++ i) {
        input[i] = static_cast<T>(i);
        expected[i] = expectedValues[i];
    }

    MinMaxNormalizationLayer<T> layer;
    Tensor<T> result = layer.forward(input);

    bool testPassed = true;
    if (result != expected) {
        std::cout << "FAIL: forwardMinMaxNormalizationLayerTest ( " << typeid(T).name() << " result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: forwardMinMaxNormalizationLayerTest\n";
    }

    return testPassed;
}

template <typename T>
bool LayerTestSuite<T>::backwardMinMaxNormalizationLayerTest() {
    std::vector <int> dimensions = {2, 2, 2};

    Tensor<T> input (dimensions.size(), dimensions);
    Tensor<T> gradient(dimensions.size(), dimensions);
    Tensor<T> expected (dimensions.size(), dimensions);
    T expectedValue = static_cast<T>(1.0/7.0);

    for (size_t i = 0; i < input.getSize(); ++ i) {
        input[i] = static_cast<T>(i);
        gradient[i] = static_cast<T> (1);
        expected[i] = expectedValue;
    }

    MinMaxNormalizationLayer<T> layer;
    layer.forward(input);
    Tensor<T> result = layer.backward(gradient, 0.0);

    bool testPassed = true;
    if (result != expected) {
        std::cout << "FAIL: backwardMinMaxNormalizationLayerTest ( " << typeid(T).name() << " result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: backwardMinMaxNormalizationLayerTest\n";
    }

    return testPassed;
 
}

//==================================================================
//  NEURAL NETWORK TESTS
//==================================================================

template class NeuralNetworkTestSuite<double>;
template class NeuralNetworkTestSuite<float>;

static const int TOTAL_NUMBER_OF_NEURAL_NETWORK_TESTS = 3;

template <typename T>
NeuralNetworkTestSuite<T>::NeuralNetworkTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_NEURAL_NETWORK_TESTS;
}

template <typename T>
void NeuralNetworkTestSuite<T>::run() {

    std::cout << "\n\nRUNNING NEURAL NETWORK TEST SUITE:\n";
    std::cout << "Type: " << typeid(T).name() << "\n";

    if(updateWeightsAndBiasesTest()) numPassed ++;
    if(forwardNetworkTest()) numPassed ++;
    if(backwardNetworkTest()) numPassed ++;

    assert(numPassed == numTests);
}

template <typename T>
bool NeuralNetworkTestSuite<T>::updateWeightsAndBiasesTest() {
    NeuralNetwork<T> network;
    const int size = 2;
    
    network.addLayer(std::make_unique<ConvolutionalLayer<T>>(1,1,1,size));
    network.addLayer(std::make_unique<FlattenLayer<T>>());
    network.addLayer(std::make_unique<DenseLayer<T>>(1, size));

    std::vector<Tensor<T>> weights;
    std::vector<Tensor<T>> biases;

    Tensor<T> convWeights (4, {size, 1, 1, 1});
    Tensor<T> convBias (1, {size});
    Tensor<T> denseWeights (2, {1, size});
    Tensor<T> denseBias (1, {size});

    for (size_t i = 0; i < convWeights.getSize(); ++ i) {
        convWeights[i] = static_cast<T>(1);
    }

    for (size_t i = 0; i < convBias.getSize(); ++ i) {
        convBias[i] = static_cast<T>(0);
    }

    for (size_t i = 0; i < denseWeights.getSize(); ++ i) {
        denseWeights[i] = static_cast<T>(1);
    }

    for (size_t i = 0; i < denseBias.getSize(); ++ i) {
        denseBias[i] = static_cast<T>(0);
    }


    weights.push_back(convWeights);
    weights.push_back(denseWeights);

    biases.push_back(convBias);
    biases.push_back(denseBias);

    network.loadWeights(weights);
    network.loadBiases(biases);

    bool testPassed = true;
    if (network.getWeights() != weights) {
        std::cout << "FAIL: updateWeightsAndBiasesTest ( " << typeid(T).name() << " weights doesnt match expected)\n";
        testPassed = false;
    }

    if (network.getBiases() != biases) {
        std::cout << "FAIL: updateWeightsAndBiasesTest ( " << typeid(T).name() << " biases doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: updateWeightsAndBiasesTest\n";
    }

    return testPassed;
}

template <typename T>
bool NeuralNetworkTestSuite<T>::forwardNetworkTest() {
    NeuralNetwork<T> network;
    const int size = 2;
    
    network.addLayer(std::make_unique<ConvolutionalLayer<T>>(1,1,1,size));
    network.addLayer(std::make_unique<MinMaxNormalizationLayer<T>>());
    network.addLayer(std::make_unique<FlattenLayer<T>>());
    network.addLayer(std::make_unique<DenseLayer<T>>(4, size));

    std::vector<Tensor<T>> weights;
    std::vector<Tensor<T>> biases;

    Tensor<T> convWeights (4, {size, 1, 1, 1});
    Tensor<T> convBias (1, {size});
    Tensor<T> denseWeights (2, {4, size});
    Tensor<T> denseBias (1, {size});

    for (size_t i = 0; i < convWeights.getSize(); ++ i) {
        convWeights[i] = static_cast<T>(1);
    }

    for (size_t i = 0; i < convBias.getSize(); ++ i) {
        convBias[i] = static_cast<T>(0);
    }

    for (size_t i = 0; i < denseWeights.getSize(); ++ i) {
        denseWeights[i] = static_cast<T>(1);
    }

    for (size_t i = 0; i < denseBias.getSize(); ++ i) {
        denseBias[i] = static_cast<T>(0);
    }


    weights.push_back(convWeights);
    weights.push_back(denseWeights);

    biases.push_back(convBias);
    biases.push_back(denseBias);

    network.loadWeights(weights);
    network.loadBiases(biases);


    Tensor<T> expectedOutput (2, {1,size});
    expectedOutput[0] = static_cast<T>(2);
    expectedOutput[1] = static_cast<T>(2);
    Tensor<T> input(4, {1, 1, 1, size});
    input[0] = static_cast<T>(0);
    input[1] = static_cast<T>(2);

    Tensor<T> actualOutput = network.forward(input);

    bool testPassed = true;
    if(actualOutput != expectedOutput) {
        std::cout << "FAIL: forwardNetworkTest ( " << typeid(T).name() << " result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: forwardNetworkTest\n";
    }

    return testPassed;
}

template <typename T>
bool NeuralNetworkTestSuite<T>::backwardNetworkTest() {

    NeuralNetwork<T> network;
    const int size = 2;
    
    network.addLayer(std::make_unique<ConvolutionalLayer<T>>(1,1,1,size));
    network.addLayer(std::make_unique<MinMaxNormalizationLayer<T>>());

    Tensor<T> convWeights (4, {size, 1, 1, 1});
    Tensor<T> convBias (1, {size});

    for (size_t i = 0; i < convWeights.getSize(); ++ i) {
        convWeights[i] = static_cast<T>(1);
    }

    for (size_t i = 0; i < convBias.getSize(); ++ i) {
        convBias[i] = static_cast<T>(0);
    }
    
    std::vector<Tensor<T>> weights;
    weights.push_back(convWeights);

    std::vector<Tensor<T>> bias;
    bias.push_back(convBias);

    network.loadWeights(weights);
    network.loadBiases(bias);

    Tensor<T> input(4, {1, 1, 1, size});
    input[0] = static_cast<T>(0);
    input[1] = static_cast<T>(2);

    Tensor<T> gradient(4, {1, size, 1, 2});
    for (size_t i = 0; i < gradient.getSize(); ++ i) {
        gradient[i] = static_cast<T>(0.5);
    }

    network.forward(input);
    const T learningRate = 1.0;
    network.backPropagate(gradient, learningRate);

    Tensor<T> expectedWeights (4, {size, 1, 1, 1});
    Tensor<T> expectedBias (1, {size});

    for(size_t i = 0; i < expectedWeights.getSize(); ++ i) {
        expectedWeights[i] = static_cast<T>(0.5);
    }
    for(size_t i = 0; i < expectedBias.getSize(); ++ i) {
        expectedBias[i] = static_cast<T>(-0.5);
    }

    bool testPassed = true;

    if (network.getWeights()[0] != expectedWeights) {
        std::cout << "FAIL: backwardNetworkTest ( " << typeid(T).name() << " weights doesnt match expected)\n";
        testPassed = false;
    }

    if (network.getBiases()[0] != expectedBias) {
        std::cout << "FAIL: backwardNetworkTest ( " << typeid(T).name() << " bias doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: backwardNetworkTest\n";
    }

    return testPassed;
}

//==================================================================
//  LOSS FUNCTION TESTS
//==================================================================

template class LossFunctionTestSuite<double>;
template class LossFunctionTestSuite<float>;

static int TOTAL_NUMBER_OF_LOSS_FUNCTION_TESTS = 1;

template <typename T>
LossFunctionTestSuite<T>::LossFunctionTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_LOSS_FUNCTION_TESTS;
}

template <typename T>
void LossFunctionTestSuite<T>::run() {
    
    std::cout << "\n\nRUNNING LOSS FUNCTION TEST SUITE:\n";
    std::cout << "Type: " << typeid(T).name() << "\n";

    if(gradientCalculationTest()) numPassed ++;

    assert(numPassed == numTests);
}

template <typename T>
bool LossFunctionTestSuite<T>::gradientCalculationTest() {
    std::vector<int> dimension = {1, 1, 2};

    Tensor<T> input (dimension.size(), dimension);
    Tensor<T> given (dimension.size(), dimension);
    Tensor<T> expected (dimension.size(), dimension);

    for (size_t i = 0; i < input.getSize(); ++ i) {
        input[i] = static_cast<T>(0.5);
        given[i] = static_cast<T>(0.25);
        expected[i] = static_cast<T>(0.25);
    }

    Tensor<T> gradient = LossFunctionCalculator::calculateGradient<T>(input, given);

    bool testPassed = true;
    if (gradient != expected) {
        std::cout << "FAIL: gradientCalculationTest ( " << typeid(T).name() << " gradient doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: gradientCalculationTest\n";
    }

    return testPassed;
}

//==================================================================
//  IMAGE ENCODER TESTS
//==================================================================
template class ImageEncoderTestSuite<double>;
template class ImageEncoderTestSuite<float>;

static int TOTAL_NUMBER_OF_IMAGE_ENCODER_TESTS = 1;

template <typename T>
ImageEncoderTestSuite<T>::ImageEncoderTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_IMAGE_ENCODER_TESTS;
}

template <typename T>
void ImageEncoderTestSuite<T>::run() {
    
    std::cout << "\n\nRUNNING IMAGE ENCODER TEST SUITE:\n";
    std::cout << "Type: " << typeid(T).name() << "\n";

    if(imageEncoderTest()) numPassed ++;

    assert(numPassed == numTests);
}

template <typename T>
bool ImageEncoderTestSuite<T>::imageEncoderTest() {
    const int numChannels = 3;
    const int height = 1600;
    const int width = 2400;
    std::vector<int> expectedDimensions = {1, numChannels, height, width};

    ImageEncoder<T> encoder;

    Tensor<T> output = encoder.encodeImage("data/tests/test.jpg");

    T expectedRedPixel [3] = {240, 0, 1};
    T expectedBlackPixel [3] = {0, 0, 0};
    T expectedWhitePixel [3] = {255, 255, 255};

    int redPixelOffset = 300 * width + 1050; // coordinates are 1050, 300
    int blackPixelOffset = 800 * width + 1200; // coordinates are 1200, 800
    int whitePixelOffset = 100 * width + 2300; // coordinates are 2300, 100

    bool testPassed = true;
    if (output.getDimension() != expectedDimensions) {
        std::cout << "FAIL: imageEncoderTest (output tensor dimensions do not equal expected)\n";
        testPassed = false;
    }

    for (int channel = 0; channel < 3; ++ channel) {
        const int channelOffset = channel * height * width;
        if (output[channelOffset + redPixelOffset] != expectedRedPixel[channel]) {
            std::cout << "FAIL: imageEncoderTest (red pixel in channel " << channel << " doesnt match expected)\n";            testPassed = false;
            std::cout << "Expected: " << expectedRedPixel[channel] << ", Actual: " << output[channelOffset + redPixelOffset] << "\n";
            testPassed = false;
        }

        if (output[channelOffset + blackPixelOffset] != expectedBlackPixel[channel]) {
            std::cout << "FAIL: imageEncoderTest (black pixel in channel " << channel << " doesnt match expected)\n";
            std::cout << "Expected: " << expectedBlackPixel[channel] << ", Actual: " << output[channelOffset + blackPixelOffset] << "\n";
            testPassed = false;
        }

        if (output[channelOffset + whitePixelOffset] != expectedWhitePixel[channel]) {
            std::cout << "FAIL: imageEncoderTest (white pixel in channel " << channel << " doesnt match expected)\n";
            std::cout << "Expected: " << expectedWhitePixel[channel] << ", Actual: " << output[channelOffset + whitePixelOffset] << "\n";
            testPassed = false;
        }
    }

    if (testPassed) {
        std::cout<< "PASS: imageEncoderTest\n";
    }

    return testPassed;
}