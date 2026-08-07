#include <iostream>
#include "testSuite.hpp"
#include "layer.hpp"

int main (int argc, char ** argv) {
    TensorTestSuite test;
    test.run();
    
    Tensor<double> t (1, {1});
    ActivationLayer<double> layer([](double x){std::cout << "Successful Init!\n"; return x;});
    layer.forward(t);
    layer.backward(t, 0);
    return 0;
}