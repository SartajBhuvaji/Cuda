#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5

__global__ void normalizeRGBImages(unsigned char* d_images, float* d_images_float,
    unsigned char* d_labels, float* d_labels_float,
    int width, int height, int numImages) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height * 3 * numImages;

    if (idx < totalPixels) {
        // Normalize the pixel value (0 to 1)
        d_images_float[idx] = static_cast<float>(d_images[idx]) / 255.0f;

        // Process labels
        int pixelsPerImage = width * height * 3;
        if (idx % pixelsPerImage == 0) {
            int imageIdx = idx / pixelsPerImage;
            if (imageIdx < numImages) {
                d_labels_float[imageIdx] = static_cast<float>(d_labels[imageIdx]);
            }
        }
    }
}



void preprocessImage(unsigned char* d_images, float** d_images_float,
    unsigned char* d_labels, float** d_labels_float)
{
    int totalPixels = IMG_SIZE * NUM_IMAGES * DATA_BATCHES; // Total number of pixels in all images
    int blockSize = 256;
    int gridSize = (totalPixels + blockSize - 1) / blockSize;

    // Allocate device memory for float arrays
    cudaMalloc(d_images_float, totalPixels * sizeof(float));
    cudaMalloc(d_labels_float, NUM_IMAGES * DATA_BATCHES * sizeof(float));

    normalizeRGBImages << <gridSize, blockSize >> > (d_images, *d_images_float, d_labels, *d_labels_float,
        32, 32, NUM_IMAGES * DATA_BATCHES);

    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in preprocessImages: %s\n", cudaGetErrorString(error));
    }

    printf("Preprocessing complete\n");
}