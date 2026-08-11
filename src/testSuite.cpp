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

static const int TOTAL_NUMBER_OF_LAYER_TESTS = 3;

LayerTestSuite::LayerTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_LAYER_TESTS;
}

void LayerTestSuite::run() {
    std::cout << "\n\nRUNNING LAYER TEST SUITE:\n\n";
    assert (numTests == numPassed);
}