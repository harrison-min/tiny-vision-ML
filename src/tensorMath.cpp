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
Tensor<T> Tensor<T>::operator*(const Tensor<T> & rhs) {
    assert(order == rhs.getOrder() && order >= 2);

    int numMatrices = 1;
    std::vector<int> outputDimension(order);

    for (int i = 0; i < order-2; ++ i) {
        assert(dimension[i] == rhs.dimension[i]);
        numMatrices *= dimension[i];
        outputDimension[i] = dimension[i];
    }

    assert(dimension[order - 1] == rhs.dimension[order-2]);
    int M = dimension[order - 2];
    int N = rhs.dimension[order - 1];
    int K = dimension[order - 1];

    outputDimension[order - 2] = M;
    outputDimension[order - 1] = N;

    int lhsOffset = M * K;
    int rhsOffset = K * N;
    int outputOffset = M * N;

    Tensor<T> output (order, outputDimension);

    for (int i = 0; i < numMatrices; ++ i) {
        for (int row = 0; row < M; ++ row) {
            for (int col = 0; col < N; ++ col)  {
                T sum = 0;
                for (int k = 0; k < K; ++ k) {
                    int lhsIndex = row * K + k;
                    int rhsIndex = k * N + col; 

                    sum += data[lhsOffset * i + lhsIndex] * rhs[rhsOffset * i + rhsIndex];
                }

                int outputIndex = row * N + col;
                output.data[outputOffset * i + outputIndex] = sum;
            }
        }
    }

    return output;
}

template<typename T>
const T& Tensor<T>::operator[](int index) const{
    return data[index];
}

template<typename T>
T& Tensor<T>::operator[](int index){
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

template<typename T>
void Tensor<T>::print() const {
    if (order < 2) {
        for (size_t i = 0; i < dataSize; ++ i) {
            std::cout << data[i] << ", ";
        }
        std::cout << "\n";
    } else {
        int M = dimension[order - 2];
        int N = dimension[order - 1];
        int offset = M * N;

        int numMatrices = 1;
        
        for (int i = 0; i < order - 2; i++) {
            numMatrices *= dimension[i];
        }

        for (int i = 0; i < numMatrices; ++ i) {
            std::cout << "\nMatrix " << i << "\n";
            for (int j = 0; j < M; ++ j) {
                for (int k = 0; k < N; ++ k ) {
                    std::cout << data [offset * i + (j * N) + k] <<  ", ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
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
