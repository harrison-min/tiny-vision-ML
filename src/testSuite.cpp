#include "testSuite.hpp"
#include "layer.hpp"
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

static const int TOTAL_NUMBER_OF_LAYER_TESTS = 4;
template class LayerTestSuite<double>;
template class LayerTestSuite<float>;

template <typename T>
LayerTestSuite<T>::LayerTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_LAYER_TESTS;
}

template <typename T>
void LayerTestSuite<T>::run() {
    std::cout << "\n\nRUNNING TENSOR TEST SUITE:\n";
    std::cout << "Type: " << typeid(T).name() << "\n";
    if (forwardDenseLayerTest()) numPassed ++;
    if (backwardDenseLayerTest()) numPassed ++;
    if (forwardActivationLayerTest()) numPassed ++;
    if (backwardActivationLayerTest()) numPassed ++;
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
    Tensor<T> TOutput = layer.forward(input);

    bool testPassed = true;
    if (TOutput != expectedOutput) {
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