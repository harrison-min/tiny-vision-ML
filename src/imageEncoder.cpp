#include "imageEncoder.hpp"
#include <iostream>
#include <cassert>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
    
static const int REQUESTED_CHANNELS = 3;

template <typename T>
Tensor<T> ImageEncoder<T>::encodeImage(const std::string & filePath) {
    int width, height, channels;

    unsigned char * data = stbi_load(filePath.c_str(), &width, &height, &channels, REQUESTED_CHANNELS); 

    assert(data != nullptr);

    std::cout << "Width: " << width << ", Height: " << height << ", Channels: " << channels << "\n";
    
    Tensor<T> output(1, {1});

    return output;
}

template class ImageEncoder<double>;
template class ImageEncoder<float>;