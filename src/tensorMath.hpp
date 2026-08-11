#pragma once
#include <vector>
#include <functional>

template<typename T>
class Tensor {
    private:
        std::vector<int> dimension;
        int order;
        size_t dataSize;
        std::vector<T> data; 

    public:
        Tensor(int tempOrder, const std::vector<int>& tempDimension);
        Tensor<T> transpose();
        Tensor<T> apply(std::function <T(T)> f) const;
        Tensor<T> collapse (const std::vector<int>& collapsingDimIndex, std::function <T(T, T)> f, T initValue);

        Tensor<T> operator+(const Tensor<T> & rhs) const;
        Tensor<T> operator-(const Tensor<T> & rhs) const;
        Tensor<T> operator*(double rhs) const;
        Tensor<T> operator*(const Tensor<T> & rhs) const;
        const T& operator[](int index) const;
        T& operator[](int index) ;
        bool operator==(const Tensor<T> & rhs) const;
        bool operator!=(const Tensor<T> & rhs) const;



        size_t getSize() const;
        int getOrder() const;
        std::vector<int> getDimension() const;
        void print() const;
};

namespace TensorCalculator {
    template <typename T>
    double innerProduct(const Tensor<T>& tensor1, const Tensor<T>& tensor2);
    
    template<typename T>
    Tensor<T> hadamardProduct(const Tensor<T>& tensor1, const Tensor<T>& tensor2);

    template <typename T>
    T reLU(T input);

    template<typename T>
    T sum(T n1, T n2);
};
