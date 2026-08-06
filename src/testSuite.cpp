#include "testSuite.hpp"
#include <iostream>
#include <limits>

static const int TOTAL_NUMBER_OF_TESTS = 8;
static const double doubleEpsilon = std::numeric_limits<double>::epsilon() * 10;
static const float floatEpsilon = std::numeric_limits<double>::epsilon() * 10;

TensorTestSuite::TensorTestSuite() {
    numPassed = 0;
    numTests = TOTAL_NUMBER_OF_TESTS;
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

    Tensor<double> zeroDoubleTensor = generateTestTensor<double>(dimension);
    Tensor<float> zeroFloatTensor = generateTestTensor<float>(dimension);

    double expectedDouble = 0;
    float expectedFloat = 0;
    
    int size = static_cast<int>(doubleTensor.getSize());
    for (int i = 0; i < size; ++ i) {
        zeroDoubleTensor[i] = 0;
        zeroFloatTensor[i] = 0;
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

    if (std::abs(TensorCalculator::innerProduct(doubleTensor, zeroDoubleTensor)) > doubleEpsilon) {
        std::cout << "FAIL: innerProductTest (Inner Product of float Tensor with 0 tensor isnt 0)\n";
        testPassed = false;
    }

    if (std::abs(TensorCalculator::innerProduct(floatTensor, zeroFloatTensor)) > floatEpsilon) {
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
    Tensor<double> zeroDoubleTensor = generateTestTensor<double>(dimension);
    Tensor<float> expectedFloatTensor = generateTestTensor<float>(dimension);
    Tensor<float> zeroFloatTensor = generateTestTensor<float>(dimension);

    int size = static_cast<int>(doubleTensor.getSize());
    for (int i = 0; i < size; ++ i) {
        expectedDoubleTensor[i] = 2 * doubleTensor[i];
        expectedFloatTensor[i] = 2 * floatTensor[i];
        zeroDoubleTensor[i] = 0;
        zeroFloatTensor[i] = 0;
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

    if (doubleTensor + zeroDoubleTensor != doubleTensor) {
        std::cout << "FAIL: matrixAdditionTest (double tensor added to 0 tensor doenst equal itself)\n";
        testPassed = false;
    }

    if (floatTensor + zeroFloatTensor != floatTensor) {
        std::cout << "FAIL: matrixAdditionTest (float tensor added to 0 tensor doenst equal itself)\n";
        testPassed = false;
    }

    if (testPassed) {
        std::cout << "PASS: matrixAdditionTest";
    }

    return testPassed;
	
}

bool TensorTestSuite::matrixMultiplicationTest(){
	bool testPassed = true;
    return testPassed;
}

bool TensorTestSuite::hadamardMultiplicationTest(){
	bool testPassed = true;
    return testPassed;
	
}

bool TensorTestSuite::biasAdditionTest(){
	bool testPassed = true;
    return testPassed;
	
}

bool TensorTestSuite::ReLUTest(){
	bool testPassed = true;
    return testPassed;
	
}

bool TensorTestSuite::sumCollapseTest(){
	bool testPassed = true;
    return testPassed;
	
}

void TensorTestSuite::run() {
    if (innerProductTest()) numPassed ++;
    if (transposeTest()) numPassed ++;
    if (matrixAdditionTest()) numPassed ++;
    if (matrixMultiplicationTest()) numPassed ++;
    if (hadamardMultiplicationTest()) numPassed ++;
    if (biasAdditionTest()) numPassed ++;
    if (ReLUTest()) numPassed ++;
    if (sumCollapseTest()) numPassed ++;

    assert(numPassed == numTests);
}
