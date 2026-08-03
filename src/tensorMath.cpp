#include "tensorMath.hpp"
#include <iostream>
#include <cassert>

template <typename T>
Tensor<T>::Tensor(int tempOrder, std::vector<int> tempDimension) {
    assert(tempOrder >= 0);

    order = tempOrder;
    dimension = tempDimension;
    dataSize = 1;
    for (int i = 0; i < order; ++ i) {
        dataSize *= dimension [i];
    }
    data.resize(dataSize);
}


template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> & rhs) {
    assert (dataSize == rhs.getSize() && order == rhs.getOrder());

    Tensor<T> sum (order, dimension);
    for (size_t i = 0; i < dataSize; ++i) {
        sum.data[i] = data [i] + rhs[i];
    }

    return sum;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(double rhs){
    Tensor<T> product (order, dimension);
    for (size_t i = 0; i < dataSize; ++i) {
        product.data[i] = data [i] * rhs;
    }

    return product;
}

template<typename T>
const T& Tensor<T>::operator[](int index) const{
    return data[index];
}

template<typename T>
size_t Tensor<T>::getSize() const{
    return dataSize;
}

template<typename T>
int Tensor<T>::getOrder() const{
    return order;
}

template class Tensor<double>;
template class Tensor<float>;

template <typename T>
double TensorCalculator::innerProduct(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
    assert(tensor1.getSize() == tensor2.getSize() && tensor1.getOrder() == tensor2.getOrder());
    double product = 0;
    
    size_t size = tensor1.getSize();
    for (size_t i = 0; i < size; ++ i) {
        product += (tensor1[i] * tensor2[i]);
    }

    return product;
}

namespace TensorCalculator {
    template double innerProduct(const Tensor<double>& tensor1, const Tensor<double>& tensor2);
    template double innerProduct(const Tensor<float>& tensor1, const Tensor<float>& tensor2);
}
