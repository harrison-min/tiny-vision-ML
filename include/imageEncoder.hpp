#pragma once
#include "tensorMath.hpp"
#include <string>

template <typename T>
class ImageEncoder {
    public:
        Tensor<T> encodeImage (const std::string & filePath);
};