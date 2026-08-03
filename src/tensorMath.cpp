#include "tensorMath.hpp"
#include <iostream>

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> & rhs) {
    std::cout << "Hello World";

    return *this;
}

template class Tensor<double>;
template class Tensor<float>;