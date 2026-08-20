#include <iostream>
#include "testSuite.hpp"
#include "imageEncoder.hpp"

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

    LossFunctionTestSuite<double> dLossTest;
    LossFunctionTestSuite<float> fLossTest;
    dLossTest.run();
    fLossTest.run();

    ImageEncoder<double> encoder;
    encoder.encodeImage("data/test_image-2532858414.jpg");
    
    return 0;
}