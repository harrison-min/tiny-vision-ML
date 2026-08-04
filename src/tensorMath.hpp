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
        Tensor<T> apply(std::function <T(T)> f);

        Tensor<T> operator+(const Tensor<T> & rhs);
        Tensor<T> operator*(double rhs);
        Tensor<T> operator*(const Tensor<T> & rhs);
        const T& operator[](int index) const;
        T& operator[](int index);

        size_t getSize() const;
        int getOrder() const;
        int getDimension(int index) const;
        void print() const;
};

namespace TensorCalculator {
    template <typename T>
    double innerProduct(const Tensor<T>& tensor1, const Tensor<T>& tensor2);
    
    template<typename T>
    Tensor<T> hadamardProduct(const Tensor<T>& tensor1, const Tensor<T>& tensor2);

    template <typename T>
    T reLU(T input);
};
