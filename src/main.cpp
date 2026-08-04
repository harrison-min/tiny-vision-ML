#include <iostream>
#include "tensorMath.hpp"
#include <random>

int main (int argc, char ** argv) {
    int column = 2;
    std::vector <int> dimensions = {2, 2, column};
    Tensor<double> bias (1, {column});

    Tensor<double> t1 (3, dimensions);
    Tensor<double> t2 (3, dimensions);

    std::uniform_real_distribution<double> unif(-1, 1);
    std::default_random_engine re;

    for (int i = 0; i < 8; ++ i) {
        t1[i] = unif(re);
        t2[i] = unif(re);
    }

    for (int i = 0; i < column; ++ i) {
        bias[i] = unif(re);
    }

    std::cout << "T1 Before\n";
    t1.print();

    std::cout << "T2 Before\n";
    t2.print();

    std::cout << "Euclidean Inner Product\n";
    std::cout << TensorCalculator::innerProduct(t1, t2) << "\n";

    std::cout << "Transpose of t2\n";
    Tensor<double> t3 = t2.transpose();
    t3.print();

    std::cout << "Matrix Addition\n";
    Tensor<double> t4 = t1 + t2;
    t4.print();

    Tensor<double> t5 = t1 * t2;
    std::cout << "Matrix Mult\n";
    t5.print();

    Tensor<double> t6 = TensorCalculator::hadamardProduct(t1, t2);
    std::cout << "Haadamard Mult\n";
    t6.print();

    Tensor<double> t7 = t1 + bias;
    std::cout << "Bias addition\n";
    t7.print();

    Tensor<double> t8 = t1.apply(TensorCalculator::reLU<double>);
    std::cout << "reLU application\n";
    t8.print();

    Tensor<double> t9 = t1.collapse({2}, TensorCalculator::sum<double>, 0);
    std::cout << "Sum collapse\n";
    t9.print();
    
    return 0;
}