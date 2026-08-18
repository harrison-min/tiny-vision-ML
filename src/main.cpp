#include <iostream>
#include "testSuite.hpp"

int main (int argc, char ** argv) {
    TensorTestSuite<double> dTensorTest;
    TensorTestSuite<float> fTensorTest;
    dTensorTest.run();
    fTensorTest.run();

    LayerTestSuite<double> dLayerTest;
    LayerTestSuite<float> fLayerTest;
    dLayerTest.run();
    fLayerTest.run();

    NeuralNetworkTestSuite<double> dNetworkTest;
    NeuralNetworkTestSuite<float> fNetworkTest;
    dNetworkTest.run();
    fNetworkTest.run();
    
    return 0;
}