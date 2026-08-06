#pragma once
#include "tensorMath.hpp"
#include <vector>
#include <cassert>

class TestSuite {
    protected:
        int numTests;
        int numPassed;
    public:
        virtual void run() = 0;
        virtual ~TestSuite() = default;
};

class TensorTestSuite : public TestSuite{
    private:
        template <typename T>
        Tensor<T> generateTestTensor(const std::vector<int> & dimension);
        bool innerProductTest();
        bool transposeTest();
        bool matrixAdditionTest();
        bool matrixMultiplicationTest();
        bool hadamardMultiplicationTest();
        bool biasAdditionTest();
        bool ReLUTest();
        bool sumCollapseTest();
    public:
        void run() override;
        TensorTestSuite();
};

