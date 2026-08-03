#include <vector>

template<typename T>
class Tensor {
    private:
        int dimension;
        std::vector<T> data; 

    public:
        Tensor<T> operator+(const Tensor<T> & rhs);
        Tensor<T> operator*(double rhs);
};

namespace TensorCalculator {
    template <typename T>
    double innerProduct(const Tensor<T>& tensor1, const Tensor<T>& tensor2);
};

