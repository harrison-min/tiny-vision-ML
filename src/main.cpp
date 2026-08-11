#include <iostream>
#include "testSuite.hpp"

int main (int argc, char ** argv) {
    TensorTestSuite tensorTest;
    tensorTest.run();

    LayerTestSuite layerTest;
    layerTest.run();
    
    return 0;
}