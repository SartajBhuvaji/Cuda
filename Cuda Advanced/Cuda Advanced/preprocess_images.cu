#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5

__global__ void convertToGrayscaleAndNormalize(unsigned char* d_images, float* d_images_gray_norm,
    unsigned char* d_labels, float* d_labels_float,
    int width, int height, int numImages) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * numImages) {
        int imageIdx = idx / (width * height);
        int pixelIdx = idx % (width * height);

        // Convert to grayscale and normalize
        float r = d_images[3 * idx] / 255.0f;
        float g = d_images[3 * idx + 1] / 255.0f;
        float b = d_images[3 * idx + 2] / 255.0f;

        // Using the luminosity method
        float gray = 0.21f * r + 0.72f * g + 0.07f * b;

        d_images_gray_norm[idx] = gray;

        // Convert labels to float (only once per image)
        if (pixelIdx == 0) {
            d_labels_float[imageIdx] = static_cast<float>(d_labels[imageIdx]);
        }
    }
}

void preprocessImages(unsigned char* d_images, float* d_images_gray_norm,
    unsigned char* d_labels, float* d_labels_float) {
    int totalPixels = IMG_SIZE / 3 * NUM_IMAGES * DATA_BATCHES;
    int blockSize = 256;
    int gridSize = (totalPixels + blockSize - 1) / blockSize;

    convertToGrayscaleAndNormalize << <gridSize, blockSize >> > (d_images, d_images_gray_norm,
        d_labels, d_labels_float,
        32, 32, NUM_IMAGES * DATA_BATCHES);
    cudaDeviceSynchronize();
}