#include <iostream>
#include "tensorMath.hpp"

int main (int argc, char ** argv) {
    std::cout << "Hello World!";
    Tensor<double> t1 (1, {1}), t2(1,{1}), t3 (2, {1,1});

    return 0;
}