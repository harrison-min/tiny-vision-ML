#include "tensorMath.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>

template <typename T>
Tensor<T>::Tensor(int tempOrder, const std::vector<int>& tempDimension) {
    assert(tempOrder >= 0);

    order = tempOrder;
    dimension = tempDimension;
    dataSize = 1;
    for (int i = 0; i < order; ++ i) {
        dataSize *= dimension [i];
    }
    data.resize(dataSize);
}

template <typename T>
Tensor<T> Tensor<T>::transpose() {
    assert(order >= 2);

    int numMatrices = 1;
    std::vector<int> outputDimension(order);

    for (int i = 0; i < order - 2; ++i) {
        numMatrices *= dimension[i];
        outputDimension[i] = dimension[i];
    }

    int M = dimension[order - 2];
    int N = dimension[order - 1];

    outputDimension[order - 2] = dimension[order - 1];
    outputDimension[order - 1] = dimension[order - 2];

    Tensor<T> output (order, outputDimension);

    int offset = M * N;

    for (int i = 0; i < numMatrices; ++ i) {
        for (int j = 0; j < M; ++ j) {
            for (int k = 0; k < N; ++ k) {
                output[offset * i + k * M + j] = data[offset * i + j * N + k];
            }
        }
    }

    return output;
}

template <typename T>
Tensor<T> Tensor<T>::apply(std::function <T(T)> f) const{
    Tensor<T> newTensor(order, dimension);
    for (size_t i = 0; i < dataSize; ++ i) {
        newTensor[i] = f(data[i]);
    }
    return newTensor;
}

template <typename T>
Tensor<T> Tensor<T>::collapse (const std::vector<int>& collapsingDimIndex, std::function <T(T, T)> f, T initValue) const{
    //calculate strides of input 
    std::vector<int> inputStrides(order);
    inputStrides[order - 1] = 1;
    for (int i = order - 2; i >=0; -- i) {
        inputStrides[i] = inputStrides[i + 1] * dimension[i + 1];
    }

    //find the dimensions we want to keep and populate the outputDimension vector
    int outputDimensionSize = order - collapsingDimIndex.size();
    assert(outputDimensionSize > 0);
    std::vector<int> outputDimension(outputDimensionSize);
    std::vector<int> keptDimensions(outputDimensionSize);

    int tempWriteIndex = 0;

    for (int i = 0; i < order; ++ i) {
        auto it = std::find(collapsingDimIndex.begin(), collapsingDimIndex.end(), i);
        if (it == collapsingDimIndex.end()) {
            outputDimension[tempWriteIndex] = dimension[i];
            keptDimensions[tempWriteIndex] = i;
            tempWriteIndex++;
        }
    }

    //create output Tensor and output strides
    Tensor<T> output (outputDimensionSize, outputDimension);

    std::vector<int> outputStrides(outputDimensionSize);
    outputStrides[outputDimensionSize - 1] = 1;
    for (int i = outputDimensionSize - 2; i >=0; -- i) {
        outputStrides[i] = outputStrides[i + 1] * outputDimension[i + 1];
    }

    for (size_t i = 0; i < output.getSize(); ++ i ) {
        output[i] = initValue;
    }

    // General algorithm: find coordinates in initial Tensor basis, 
    // find the outputIndex based on strides + cooresponding coordinates, 
    // apply function to output at the outputIndex + the data
    // Can be thought of as many to 1 mapping
    std::vector<int> coordinates(order);
    
    for (int i = 0; i < static_cast<int>(dataSize); ++ i) {
        int remainder = i;
        int outputIndex = 0;
        int outputStridesIndex = 0;
        for (int j = 0; j < order; ++j) {
            coordinates [j] = remainder/inputStrides[j];
            remainder %= inputStrides[j];

            if (outputStridesIndex <outputDimensionSize && j == keptDimensions[outputStridesIndex]) {
                outputIndex += (coordinates[j] * outputStrides[outputStridesIndex]);
                outputStridesIndex ++;
            }
        }

        output[outputIndex] = f(output[outputIndex], data[i]);

    }
   
    return output;
}

//==================================================================
//  OPERATOR OVERLOADING
//==================================================================


template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> & rhs) const {
    assert ((dataSize == rhs.getSize() && order == rhs.getOrder()) || 
        (dimension[order - 1] == rhs.dimension[0] && rhs.getOrder() == 1));

    Tensor<T> sum (order, dimension);
    if (dataSize == rhs.getSize() && order == rhs.getOrder()) {
        for (size_t i = 0; i < dataSize; ++i) {
            sum[i] = data [i] + rhs[i];
        }
    } else {
        int M = dimension[order - 2];
        int N = dimension[order - 1];
        int offset = M * N;
        int numMatrices = 1;

        for (int i = 0; i < order - 2; ++i ) {
            numMatrices *= dimension[i];
        }

        for (int i = 0; i < numMatrices; ++ i) {
            for (int row = 0; row < M; ++ row) {
                for (int col = 0; col < N; ++ col){
                    int index = offset * i + row * N + col;
                    sum[index] = data[index] + rhs[col];
                }
            }
        }
    }

    return sum;
}

template<typename T> 
Tensor<T> Tensor<T>::operator-(const Tensor<T> & rhs) const {
    return *this + (rhs * static_cast<T>(-1));
}

template<typename T>
Tensor<T> Tensor<T>::operator*(double rhs) const{
    Tensor<T> product (order, dimension);
    for (size_t i = 0; i < dataSize; ++i) {
        product[i] = data [i] * rhs;
    }

    return product;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> & rhs) const{
    assert(order == rhs.getOrder() && order >= 2);

    int numMatrices = 1;
    std::vector<int> outputDimension(order);

    for (int i = 0; i < order-2; ++ i) {
        assert(dimension[i] == rhs.dimension[i]);
        numMatrices *= dimension[i];
        outputDimension[i] = dimension[i];
    }

    assert(dimension[order - 1] == rhs.dimension[order-2]);
    // Multiplying a MxK matrix by KxN matrix
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
                output[outputOffset * i + outputIndex] = sum;
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
T& Tensor<T>::operator[](int index) {
    return data[index];
}

template <typename T>
bool Tensor<T>::operator==(const Tensor<T> & rhs) const{
    static const T epsilon = std::numeric_limits<T>::epsilon() * 10;
    if (dataSize != rhs.getSize() || dimension != rhs.getDimension()) {
        return false;
    }

    for (size_t i = 0 ; i < dataSize; ++ i) {
        if (std::abs(data[i] - rhs[i]) > epsilon) {
            return false;
        }
    }

    return true;
}

template <typename T>
bool Tensor<T>::operator!=(const Tensor<T> & rhs) const{
    return !(*this == rhs);
}

//==================================================================
//  HELPER AND GETTERS
//==================================================================

template<typename T>
size_t Tensor<T>::getSize() const{
    return dataSize;
}

template<typename T>
int Tensor<T>::getOrder() const{
    return order;
}

template<typename T>
std::vector<int> Tensor<T>::getDimension() const{
    return dimension;
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

//==================================================================
//  TensorCalculator
//==================================================================

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

template <typename T>
Tensor<T> TensorCalculator::hadamardProduct(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
    assert(tensor1.getOrder() == tensor2.getOrder());
    int order = tensor1.getOrder();
    std::vector<int> outputDimension(order);
    assert(tensor1.getDimension() == tensor2.getDimension());
    outputDimension = tensor1.getDimension();

    size_t size = tensor1.getSize();
    Tensor<T> output(order, outputDimension);

    for (size_t i = 0; i < size; ++ i) {
        output[i] = tensor1[i] * tensor2[i];
    }

    return output; 
}

template<typename T>
T TensorCalculator::reLU(T input) {
    return (input + std::abs(input)) * static_cast<T>(0.5);
}

template <typename T>
T TensorCalculator::derivativeReLU (T input) {
    return static_cast<T>(input > 0.0 ? 1.0 : 0.0);
}

template<typename T>
T TensorCalculator::sum(T n1, T n2) {
    return n1 + n2;
}

namespace TensorCalculator {
    template double innerProduct(const Tensor<double>& tensor1, const Tensor<double>& tensor2);
    template double innerProduct(const Tensor<float>& tensor1, const Tensor<float>& tensor2);
    template Tensor<float> hadamardProduct(const Tensor<float>& tensor1, const Tensor<float>& tensor2);
    template Tensor<double> hadamardProduct(const Tensor<double>& tensor1, const Tensor<double>& tensor2);
    template double reLU(double input);
    template float reLU(float input);
    template double derivativeReLU(double input);
    template float derivativeReLU(float input);
    template double sum(double n1, double n2);
    template float sum(float n1, float n2);
}
