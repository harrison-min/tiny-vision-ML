#include <vector>

template<typename T>
class Tensor {
    private:
        std::vector<int> dimension;
        int order;
        size_t dataSize;
        std::vector<T> data; 

    public:
        Tensor(int tempOrder, std::vector<int> tempDimension);

        Tensor<T> operator+(const Tensor<T> & rhs);
        Tensor<T> operator*(double rhs);
        Tensor<T> operator*(const Tensor<T> & rhs);
        const T& operator[](int index) const;
        T& operator[](int index);

        size_t getSize() const;
        int getOrder() const;
        void print() const;
};

namespace TensorCalculator {
    template <typename T>
    double innerProduct(const Tensor<T>& tensor1, const Tensor<T>& tensor2);
};
