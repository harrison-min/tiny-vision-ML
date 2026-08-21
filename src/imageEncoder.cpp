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
    
    Tensor<T> output (4, {1, REQUESTED_CHANNELS, height, width});

    for (int c = 0; c < REQUESTED_CHANNELS; ++ c) {
        for (int y = 0; y < height; ++ y) {
            for (int x = 0; x < width; ++ x) {
                int dataIndex = (y * width * REQUESTED_CHANNELS) + (x * REQUESTED_CHANNELS) + c;
                int outputIndex = (c * height * width) + (y * width) + x;

                output[outputIndex] = data[dataIndex];
            }
        }
    }

    stbi_image_free(data);

    return output;
}

template class ImageEncoder<double>;
template class ImageEncoder<float>;