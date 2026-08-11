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
    
    return 0;
}