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

template <typename T>
class TensorTestSuite : public TestSuite{
    private:
        Tensor<T> generateTestTensor(const std::vector<int> & dimension);
        bool innerProductTest();
        bool transposeTest();
        bool matrixAdditionTest();
        bool matrixMultiplicationTest();
        bool hadamardProductTest();
        bool biasAdditionTest();
        bool ReLUTest();
        bool sumCollapseTest();
    public:
        void run() override;
        TensorTestSuite();
};

template <typename T>
class LayerTestSuite :public TestSuite {
        bool forwardDenseLayerTest();
        bool backwardDenseLayerTest();
        bool forwardActivationLayerTest();
        bool backwardActivationLayerTest();
        bool forwardConvolutionalLayerTest();
        bool backwardConvolutionalLayerTest();
    public: 
        LayerTestSuite();
        void run() override;
};