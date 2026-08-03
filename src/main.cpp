#include <iostream>
#include "tensorMath.hpp"
#include <random>

int main (int argc, char ** argv) {
    std::cout << "Hello World!";

    std::vector <int> dimensions = {2, 2, 2, 2};
    Tensor<double> t1 (4, dimensions);
    Tensor<double> t2 (4, dimensions);

    std::uniform_real_distribution<double> unif(0, 1);
    std::default_random_engine re;

    for (int i = 0; i < 16; ++ i) {
        t1[i] = unif(re);
        t2[i] = unif(re);
    }

    std::cout << "T1 Before\n";
    t1.print();

    std::cout << "T2 Before\n";
    t2.print();

    Tensor<double> t3 = t1 * t2;
    std::cout << "Matrix Mult\n";
    t3.print();

    return 0;
}