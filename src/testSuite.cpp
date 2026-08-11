#include "testSuite.hpp"
#include "layer.hpp"
#include <iostream>
#include <limits>

//==================================================================
//  TENSOR TESTS
//==================================================================

static const int TOTAL_NUMBER_OF_TENSOR_TESTS = 8;
static const double doubleEpsilon = std::numeric_limits<double>::epsilon() * 10;
static const float floatEpsilon = std::numeric_limits<double>::epsilon() * 10;

TensorTestSuite::TensorTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_TENSOR_TESTS;
}

void TensorTestSuite::run() {
    std::cout << "\n\nRUNNING TENSOR TEST SUITE:\n\n";
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
Tensor<T> TensorTestSuite::generateTestTensor(const std::vector<int> & dimension) {
    Tensor<T> testTensor(dimension.size(), dimension);
    int totalSize = testTensor.getSize();

    for (int i = 0; i < totalSize; ++ i) {
        testTensor[i] = static_cast<T>(i - totalSize/2);
    }

    return testTensor;
}

bool TensorTestSuite::innerProductTest(){
    std::vector<int> dimension = {3, 3, 3, 3};
    Tensor<double> doubleTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatTensor = generateTestTensor<float>(dimension);

    Tensor<double> doubleZeroTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatZeroTensor = generateTestTensor<float>(dimension);

    double expectedDouble = 0;
    float expectedFloat = 0;
    
    const int size = static_cast<int>(doubleTensor.getSize());
    for (int i = 0; i < size; ++ i) {
        doubleZeroTensor[i] = 0;
        floatZeroTensor[i] = 0;
        expectedDouble += doubleTensor[i] * doubleTensor[i];
        expectedFloat += floatTensor[i] * floatTensor[i];
    }

    bool testPassed = true;


    if (std::abs(TensorCalculator::innerProduct(doubleTensor, doubleTensor) - expectedDouble) > doubleEpsilon) {
        std::cout << "FAIL: innerProductTest (Inner Product of Double Tensor with itself doesnt match expected)\n";
        testPassed = false;
    }

    if (std::abs(TensorCalculator::innerProduct(floatTensor, floatTensor) - expectedFloat) > floatEpsilon) {
        std::cout << "FAIL: innerProductTest (Inner Product of float Tensor with itself doesnt match expected)\n";
        testPassed = false;
    }

    if (std::abs(TensorCalculator::innerProduct(doubleTensor, doubleZeroTensor)) > doubleEpsilon) {
        std::cout << "FAIL: innerProductTest (Inner Product of float Tensor with 0 tensor isnt 0)\n";
        testPassed = false;
    }

    if (std::abs(TensorCalculator::innerProduct(floatTensor, floatZeroTensor)) > floatEpsilon) {
        std::cout << "FAIL: innerProductTest (Inner Product of float Tensor with 0 tensor isnt 0)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: innerProductTest\n";
    }

    return testPassed;
}

bool TensorTestSuite::transposeTest(){
    std::vector<int> dimension = {3, 2, 2, 4};
    Tensor<double> doubleTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatTensor = generateTestTensor<float>(dimension);
 
	bool testPassed = true;

    if (doubleTensor.transpose().transpose() != doubleTensor) {
        std::cout << "FAIL: transposeTest (double Tensor Transpose of the Transpose doesnt equal the original)\n";
        testPassed = false;
    }

    if (floatTensor.transpose().transpose() != floatTensor) {
        std::cout << "FAIL: transposeTest (double Tensor Transpose of the Transpose doesnt equal the original)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: transposeTest\n";
    }

    return testPassed;
}

bool TensorTestSuite::matrixAdditionTest(){
    std::vector<int> dimension = {3, 2, 2, 4};
    Tensor<double> doubleTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatTensor = generateTestTensor<float>(dimension);

    Tensor<double> expectedDoubleTensor = generateTestTensor<double>(dimension);
    Tensor<double> doubleZeroTensor = generateTestTensor<double>(dimension);
    Tensor<float> expectedFloatTensor = generateTestTensor<float>(dimension);
    Tensor<float> floatZeroTensor = generateTestTensor<float>(dimension);

    const int size = static_cast<int>(doubleTensor.getSize());
    for (int i = 0; i < size; ++ i) {
        expectedDoubleTensor[i] = 2 * doubleTensor[i];
        expectedFloatTensor[i] = 2 * floatTensor[i];
        doubleZeroTensor[i] = 0;
        floatZeroTensor[i] = 0;
    }

	bool testPassed = true;
    if (doubleTensor + doubleTensor != expectedDoubleTensor) {
        std::cout << "FAIL: matrixAdditionTest (double tensor added to itself doenst match expected)\n";
        testPassed = false;
    }

    if (floatTensor + floatTensor != expectedFloatTensor) {
        std::cout << "FAIL: matrixAdditionTest (float tensor added to itself doenst match expected)\n";
        testPassed = false;
    }

    if (doubleTensor + doubleZeroTensor != doubleTensor) {
        std::cout << "FAIL: matrixAdditionTest (double tensor added to 0 tensor doenst equal itself)\n";
        testPassed = false;
    }

    if (floatTensor + floatZeroTensor != floatTensor) {
        std::cout << "FAIL: matrixAdditionTest (float tensor added to 0 tensor doenst equal itself)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: matrixAdditionTest\n";
    }

    return testPassed;
	
}

bool TensorTestSuite::matrixMultiplicationTest(){
 
    std::vector<int> dimension = {3, 2, 4, 4};
    const int dimensionSize = static_cast<int>(dimension.size());
    std::vector<int> stride (dimensionSize);
    stride[dimensionSize - 1] = 1;

    for (int i = dimensionSize - 2; i >=0; --i) {
        stride[i] = dimension[i + 1] * stride[i + 1];
    }

    Tensor<double> doubleTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatTensor = generateTestTensor<float>(dimension);

    Tensor<double> doubleIdentityTensor = generateTestTensor<double>(dimension);
    Tensor<double> doubleZeroTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatIdentityTensor = generateTestTensor<float>(dimension);
    Tensor<float> floatZeroTensor = generateTestTensor<float>(dimension);

    const int size = static_cast<int>(doubleTensor.getSize());

    for (int i = 0; i < size; ++ i) {
        doubleZeroTensor[i] = 0;
        floatZeroTensor[i] = 0;

        int row = (i / stride[dimensionSize - 2]) %dimension[dimensionSize - 2];
        int col = (i / stride[dimensionSize - 1]) %dimension[dimensionSize - 1];

        bool isIdentity = (row == col);

        if (isIdentity) {
            doubleIdentityTensor[i] = 1;
            floatIdentityTensor[i] = 1;
        } else {
            doubleIdentityTensor[i] = 0;
            floatIdentityTensor[i] = 0;
        }
    }


	bool testPassed = true;

    if (doubleTensor * doubleIdentityTensor != doubleTensor) {
        std::cout << "FAIL: matrixMultiplicationTest (double tensor multiplied to identity doenst match itself)\n";
        testPassed = false;
    }

    if (floatTensor * floatIdentityTensor != floatTensor) {
        std::cout << "FAIL: matrixMultiplicationTest (float tensor multiplied to identity doenst match itself)\n";
        testPassed = false;
    }

    if (doubleTensor * doubleZeroTensor != doubleZeroTensor) {
        std::cout << "FAIL: matrixMultiplicationTest (double tensor multiplied to 0 Tensor doenst equal 0 tensor)\n";
        testPassed = false;
    }

    if (floatTensor * floatZeroTensor != floatZeroTensor) {
        std::cout << "FAIL: matrixMultiplicationTest (float tensor multiplied to 0 Tensor doenst equal 0 tensor)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: matrixMultiplicationTest\n";
    }

    return testPassed;
}

bool TensorTestSuite::hadamardProductTest(){
    std::vector<int> dimension = {3, 2, 3, 4, 5};
    Tensor<double> doubleTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatTensor = generateTestTensor<float>(dimension);

    Tensor<double> doubleOneTensor = generateTestTensor<double>(dimension);
    Tensor<double> doubleZeroTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatOneTensor = generateTestTensor<float>(dimension);
    Tensor<float> floatZeroTensor = generateTestTensor<float>(dimension);
    const int size = static_cast<int>(doubleTensor.getSize());

    for (int i = 0; i < size; ++ i) {
        doubleOneTensor[i] = 1;
        floatOneTensor[i] = 1;
        doubleZeroTensor[i] = 0;
        floatZeroTensor[i] = 0;
    }

	bool testPassed = true;

    if (TensorCalculator::hadamardProduct(doubleTensor, doubleOneTensor) != doubleTensor) {
        std::cout << "FAIL: hadamardProductTest (double tensor multiplied to 1 tensor doenst match itself)\n";
        testPassed = false;
    }

    if (TensorCalculator::hadamardProduct(floatTensor, floatOneTensor) != floatTensor) {
        std::cout << "FAIL: hadamardProductTest (float tensor multiplied to 1 tensor doenst match itself)\n";
        testPassed = false;
    }

    if (TensorCalculator::hadamardProduct(doubleTensor, doubleZeroTensor) != doubleZeroTensor) {
        std::cout << "FAIL: hadamardProductTest (double tensor multiplied to 0 tensor doenst equal zero)\n";
        testPassed = false;
    }

    if (TensorCalculator::hadamardProduct(floatTensor, floatZeroTensor) != floatZeroTensor) {
        std::cout << "FAIL: hadamardProductTest (float tensor multiplied to 0 tensor doenst equal zero)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: hadamardProductTest\n";
    }

    return testPassed;
	
}

bool TensorTestSuite::biasAdditionTest(){
    const int columnLength = 5;
    std::vector<int> dimension = {3, 2, 3, 4, columnLength};
    Tensor<double> doubleTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatTensor = generateTestTensor<float>(dimension);
    
    Tensor<double> doubleExpectedTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatExpectedTensor = generateTestTensor<float>(dimension);

    const int size = static_cast<int>(doubleTensor.getSize());

    for (int i = 0; i < size; ++ i) {
        doubleExpectedTensor[i] = doubleTensor[i] + 1;
        floatExpectedTensor[i] = floatTensor[i] + 1;
    }

    std::vector<int> biasDimension = {columnLength};
    Tensor<double> doubleOneBiasVector = generateTestTensor<double>(biasDimension);
    Tensor<double> doubleZeroBiasVector = generateTestTensor<double>(biasDimension);
    Tensor<float> floatOneBiasVector = generateTestTensor<float>(biasDimension);
    Tensor<float> floatZeroBiasVector = generateTestTensor<float>(biasDimension);

    for (int i = 0; i < columnLength; ++ i) {
        doubleOneBiasVector[i] = 1; 
        floatOneBiasVector[i] = 1; 
        doubleZeroBiasVector[i] = 0;
        floatZeroBiasVector[i] = 0;
    }

	bool testPassed = true;
    if (doubleTensor + doubleOneBiasVector != doubleExpectedTensor) {
        std::cout << "FAIL: biasAdditionTest (double tensor added with bias doenst match the expected)\n";
        testPassed = false;
    }

    if (floatTensor + floatOneBiasVector != floatExpectedTensor) {
        std::cout << "FAIL: biasAdditionTest (float tensor added with bias doenst match the expected)\n";
        testPassed = false;
    }

    if (doubleTensor + doubleZeroBiasVector != doubleTensor) {
        std::cout << "FAIL: biasAdditionTest (double tensor added with 0 bias doenst match itself)\n";
        testPassed = false;
    }

    if (floatTensor + floatZeroBiasVector != floatTensor) {
        std::cout << "FAIL: biasAdditionTest (float tensor added with 0 bias doenst match itself)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout<< "PASS: biasAdditionTest\n";
    }

    return testPassed;
	
}

bool TensorTestSuite::ReLUTest(){
    std::vector<int> dimension = {3, 2, 3, 4, 7};
    Tensor<double> doubleTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatTensor = generateTestTensor<float>(dimension);
    
    Tensor<double> doubleExpectedTensor = generateTestTensor<double>(dimension);
    Tensor<float> floatExpectedTensor = generateTestTensor<float>(dimension);

    const int size = static_cast<int>(doubleTensor.getSize());

    for (int i = 0; i < size; ++ i) {
        if (doubleTensor[i] < 0) {
            doubleExpectedTensor[i] = 0;
        } else {
            doubleExpectedTensor[i] = doubleTensor[i];
        }
        if (floatTensor[i] < 0) {
            floatExpectedTensor[i] = 0;
        } else {
            floatExpectedTensor[i] = floatTensor[i];
        }
    }

	bool testPassed = true;
    if (doubleTensor.apply(TensorCalculator::reLU<double>) != doubleExpectedTensor) {
        std::cout << "FAIL: ReLUTest (ReLU application on double Tensor did not match expected)\n";
        testPassed = false;
    }

    if (floatTensor.apply(TensorCalculator::reLU<float>) != floatExpectedTensor) {
        std::cout << "FAIL: ReLUTest (ReLU application on float Tensor did not match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: ReLUTest\n";
    }

    return testPassed;
}

bool TensorTestSuite::sumCollapseTest(){
    std::vector<int> dimension = {2, 3, 4, 3, 4};
    std::vector<int> eliminatedDimensions = {3, 4};

    std::vector<int> collapsedDimensions = dimension;
    for (int i = 0; i < static_cast<int>(eliminatedDimensions.size()); ++ i) {
        collapsedDimensions.erase(collapsedDimensions.begin() + eliminatedDimensions[i] - i);
    }

    Tensor<double> doubleTensor = generateTestTensor<double>(dimension);
    Tensor<double> doubleExpectedTensor = generateTestTensor<double>(collapsedDimensions);
    Tensor<float> floatTensor = generateTestTensor<float>(dimension);
    Tensor<float> floatExpectedTensor = generateTestTensor<float>(collapsedDimensions);

    const int expectedSize = static_cast<int>(doubleExpectedTensor.getSize());
    int innerSize = 1;
    for (auto dim : eliminatedDimensions) {
        innerSize *= dimension[dim];
    }

    for (int i = 0; i < expectedSize; ++ i) {
        double doubleSum = 0;
        float floatSum = 0;
        for (int j = 0; j < innerSize; ++j) {
            doubleSum += doubleTensor[i * innerSize + j];
            floatSum += floatTensor[i * innerSize + j];
        }
        doubleExpectedTensor[i] = doubleSum;
        floatExpectedTensor[i] = floatSum;
    }

	bool testPassed = true;
    if (doubleTensor.collapse(eliminatedDimensions, TensorCalculator::sum<double>, 0) != doubleExpectedTensor) {
        std::cout << "FAIL: sumCollapseTest (Sum Collapse on double Tensor did not match expected)\n";
        testPassed = false;
    }

    if (floatTensor.collapse(eliminatedDimensions, TensorCalculator::sum<float>, 0) != floatExpectedTensor) {
        std::cout << "FAIL: sumCollapseTest (Sum Collapse on float Tensor did not match expected)\n";
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

LayerTestSuite::LayerTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_LAYER_TESTS;
}

void LayerTestSuite::run() {
    std::cout << "\n\nRUNNING LAYER TEST SUITE:\n\n";
    if (forwardDenseLayerTest()) numPassed ++;
    if (backwardDenseLayerTest()) numPassed ++;
    if (forwardActivationLayerTest()) numPassed ++;
    if (backwardActivationLayerTest()) numPassed ++;
    assert (numTests == numPassed);
}

bool LayerTestSuite::forwardDenseLayerTest() {
    const int inputSize = 3;
    const int outputSize = 2;
    const int batchSize = 5;
    DenseLayer<double> doubleLayer(inputSize, outputSize);
    DenseLayer<float> floatLayer(inputSize, outputSize);

    Tensor<double> doubleWeights (2, {inputSize, outputSize});
    Tensor<float> floatWeights (2, {inputSize, outputSize});

    Tensor<double> doubleBias (1, {outputSize});
    Tensor<float> floatBias (1, {outputSize});

    Tensor<double> doubleInput(2, {batchSize, inputSize});
    Tensor<float> floatInput(2, {batchSize, inputSize});

    Tensor<double> doubleExpectedOutput(2, {batchSize,outputSize});
    Tensor<float> floatExpectedOutput(2, {batchSize,outputSize});

    for (size_t i = 0; i < doubleWeights.getSize(); ++ i) {
        doubleWeights[i] = -1.0;
        floatWeights[i] = -1.0;
    }

    for (size_t i = 0; i < doubleBias.getSize(); ++ i) {
        doubleBias [i] = 1.5;
        floatBias [i] = 1.5;
    }

    for (size_t i = 0; i < doubleExpectedOutput.getSize(); ++ i) {
        doubleExpectedOutput[i] = -1.5;
        floatExpectedOutput[i] = -1.5;
    }

    for (size_t i = 0; i < doubleInput.getSize(); ++ i) {
        doubleInput [i] = 1.0;
        floatInput [i] = 1.0;
    }
    doubleLayer.updateBias(doubleBias);
    doubleLayer.updateWeights(doubleWeights);
    floatLayer.updateBias(floatBias);
    floatLayer.updateWeights(floatWeights);
    Tensor<double> doubleOutput = doubleLayer.forward(doubleInput);
    Tensor<float> floatOutput = floatLayer.forward(floatInput);

    bool testPassed = true;
    if (doubleOutput != doubleExpectedOutput) {
        std::cout << "FAIL: forwardDenseLayerTest (double output tensor does not match expected)\n";
        testPassed = false;
    }

    if (floatOutput != floatExpectedOutput) {
        std::cout << "FAIL: forwardDenseLayerTest (float output tensor does not match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: forwardDenseLayerTest\n";
    }

    return testPassed;
}
   
bool LayerTestSuite::backwardDenseLayerTest() {
    const int inputSize = 3;
    const int outputSize = 2;
    const int batchSize = 5;
    const float floatLearningRate = 1.0;
    const double doubleLearningRate = 1.0;

    DenseLayer<double> doubleLayer(inputSize, outputSize);
    DenseLayer<float> floatLayer(inputSize, outputSize);

    Tensor<double> doubleWeights (2, {inputSize, outputSize});
    Tensor<float> floatWeights (2, {inputSize, outputSize});

    Tensor<double> doubleBias (1, {outputSize});
    Tensor<float> floatBias (1, {outputSize});

    Tensor<double> doubleInput(2, {batchSize, inputSize});
    Tensor<float> floatInput(2, {batchSize, inputSize});

    Tensor<double> doubleGradient(2, {batchSize, outputSize});
    Tensor<float> floatGradient(2, {batchSize, outputSize});

    Tensor<double> doubleExpectedOutputGradient(2, {batchSize, inputSize});
    Tensor<double> doubleExpectedWeights(2, {inputSize, outputSize});
    Tensor<double> doubleExpectedBias (1, {outputSize});
    Tensor<float> floatExpectedOutputGradient(2, {batchSize, inputSize});
    Tensor<float> floatExpectedWeights(2, {inputSize, outputSize});
    Tensor<float> floatExpectedBias (1, {outputSize});

    for (size_t i = 0; i < doubleWeights.getSize(); ++ i) {
        doubleWeights[i] = -1.0;
        floatWeights[i] = -1.0;
    }

    for (size_t i = 0; i < doubleGradient.getSize(); ++ i) {
        doubleGradient[i] = 1.0;
        floatGradient[i] = 1.0;
    }

    for (size_t i = 0; i < doubleBias.getSize(); ++ i) {
        doubleBias [i] = 1.5;
        floatBias [i] = 1.5;
        double doubleDB = 1.0 * batchSize;
        float floatDB = 1.0 * batchSize;
        doubleExpectedBias[i] = doubleBias[i] - doubleDB * doubleLearningRate;
        floatExpectedBias[i] = floatBias[i] - floatDB * floatLearningRate;
    }

    for (size_t i = 0; i < doubleInput.getSize(); ++ i) {
        doubleInput [i] = 1.0;
        floatInput [i] = 1.0;
    }

    doubleExpectedOutputGradient = doubleGradient * doubleWeights.transpose();
    floatExpectedOutputGradient = floatGradient * floatWeights.transpose();

    Tensor<double> doubleDW = doubleInput.transpose() * doubleGradient;
    Tensor<float> floatDW = floatInput.transpose() * floatGradient;

    doubleExpectedWeights = doubleWeights - (doubleDW * doubleLearningRate);
    floatExpectedWeights = floatWeights - (floatDW * floatLearningRate);

    doubleLayer.updateBias(doubleBias);
    doubleLayer.updateWeights(doubleWeights);
    floatLayer.updateBias(floatBias);
    floatLayer.updateWeights(floatWeights);

    doubleLayer.forward(doubleInput);
    Tensor<double> doubleResultGradient = doubleLayer.backward(doubleGradient, doubleLearningRate);
    floatLayer.forward(floatInput);
    Tensor<float> floatResultGradient = floatLayer.backward(floatGradient, floatLearningRate);

    bool testPassed = true;
    if (doubleResultGradient != doubleExpectedOutputGradient) {
        std::cout << "FAIL: backwardDenseLayerTest (double gradient doesnt match expected)\n";
        testPassed = false;
    }

    if (doubleLayer.getWeights()!= doubleExpectedWeights) {
        std::cout << "FAIL: backwardDenseLayerTest (double weights doesnt match expected)\n";
        testPassed = false;
    }

    if (doubleLayer.getBias()!= doubleExpectedBias) {
        std::cout << "FAIL: backwardDenseLayerTest (double bias doesnt match expected)\n";
        testPassed = false;
    }

    if (floatResultGradient != floatExpectedOutputGradient) {
        std::cout << "FAIL: backwardDenseLayerTest (float gradient doesnt match expected)\n";
        testPassed = false;
    }

    if (floatLayer.getWeights()!= floatExpectedWeights) {
        std::cout << "FAIL: backwardDenseLayerTest (float weights doesnt match expected)\n";
        testPassed = false;
    }

    if (floatLayer.getBias()!= floatExpectedBias) {
        std::cout << "FAIL: backwardDenseLayerTest (float bias doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: backwardDenseLayerTest\n";
    }

    return testPassed;
}

bool LayerTestSuite::forwardActivationLayerTest() {
    const int inputSize = 5;
    ActivationLayer<double> doubleLayer (inputSize, TensorCalculator::reLU<double>, TensorCalculator::derivativeReLU<double>);
    ActivationLayer<float> floatLayer (inputSize, TensorCalculator::reLU<float>, TensorCalculator::derivativeReLU<float>);

    Tensor<double> doubleInput (2, {1, inputSize});
    Tensor<float> floatInput (2, {1, inputSize});
    Tensor<double> doubleExpected (2, {1, inputSize});
    Tensor<float> floatExpected (2, {1, inputSize});

    const int size = static_cast<int>(doubleInput.getSize());
    for (int i = 0; i < size; ++ i) {
        doubleInput [i] = i - size/2;
        floatInput [i] = i - size/2;
        if (i - size/2 <= 0) {
            doubleExpected[i] = 0.0;
            floatExpected[i] = 0.0;
        } else {
            doubleExpected[i] = i - size/2;
            floatExpected[i] = i - size/2;
        }
    }

    Tensor<double> doubleResult = doubleLayer.forward(doubleInput);
    Tensor<float> floatResult = floatLayer.forward(floatInput);

    bool testPassed = true;
    if (doubleResult != doubleExpected) {
        std::cout << "FAIL: forwardActivationLayerTest (double result doesnt match expected)\n";
        testPassed = false;
    }

    if (floatResult != floatExpected) {
        std::cout << "FAIL: forwardActivationLayerTest (float result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: forwardActivationLayerTest\n";
    }

    return testPassed;
}

bool LayerTestSuite::backwardActivationLayerTest() {
    const int inputSize = 5;
    ActivationLayer<double> doubleLayer (inputSize, TensorCalculator::reLU<double>, TensorCalculator::derivativeReLU<double>);
    ActivationLayer<float> floatLayer (inputSize, TensorCalculator::reLU<float>, TensorCalculator::derivativeReLU<float>);

    Tensor<double> doubleInput (2, {1, inputSize});
    Tensor<float> floatInput (2, {1, inputSize});
    Tensor<double> doubleGradient (2, {1, inputSize});
    Tensor<float> floatGradient (2, {1, inputSize});
    Tensor<double> doubleExpected (2, {1, inputSize});
    Tensor<float> floatExpected (2, {1, inputSize});

    const int size = static_cast<int>(doubleInput.getSize());
    for (int i = 0; i < size; ++ i) {
        doubleInput [i] = i - size/2;
        floatInput [i] = i - size/2;
        doubleGradient[i] = i;
        floatGradient[i] = i;
        double doubleDerivativeValue = 0.0;
        float floatDerivativeValue = 0.0;
        if (i - size/2 > 0) {
            doubleDerivativeValue = 1.0;
            floatDerivativeValue = 1.0;
        } 
        doubleExpected[i]  = doubleDerivativeValue * doubleGradient[i];
        floatExpected[i]  = floatDerivativeValue * floatGradient[i];
    }

    const double doubleLearningRate = 1.0; // these dont really apply to this layer
    const float floatLearningRate = 1.0;
    doubleLayer.forward(doubleInput);
    floatLayer.forward(floatInput);
    Tensor<double> doubleResult = doubleLayer.backward(doubleGradient, doubleLearningRate);
    Tensor<float> floatResult = floatLayer.backward(floatGradient, floatLearningRate);

    bool testPassed = true;
    if (doubleResult != doubleExpected) {
        std::cout << "FAIL: backwardActivationLayerTest (double result doesnt match expected)\n";
        testPassed = false;
    }

    if (floatResult != floatExpected) {
        std::cout << "FAIL: backwardActivationLayerTest (float result doesnt match expected)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: backwardActivationLayerTest\n";
    }

    return testPassed;
}