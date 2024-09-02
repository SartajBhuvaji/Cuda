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

    // Total number of pixels in one image (including all 3 channels)
    int numPixels = width * height * 3;

    if (idx < numPixels * numImages) {
        // Normalize the pixel value (0 to 1)
        d_images_float[idx] = static_cast<float>(d_images[idx])/ 255.0f;

        if (idx % numPixels == 0) {
            int imageIdx = idx / numPixels;  // Determine which image we're working on
            d_labels_float[imageIdx] = static_cast<float>(d_labels[imageIdx]);
        }
    }
}


__global__ void greyscaleNormalization(unsigned char* d_images, float* d_images_float,
    unsigned char* d_labels, float* d_labels_float,
    int width, int height, int numImages) {

	// TO FIX : This kernel convers the image to 12*12*1 instead of 32*32*1

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of pixels in one grayscale image
    int numPixels = width * height;

    if (idx < numPixels * numImages) {
        int imageIdx = idx / numPixels;  // Determine which image we're working on
        int pixelIdx = idx % numPixels;  // Determine the pixel's index within that image

        int rgbPixelIdx = imageIdx * (numPixels * 3) + pixelIdx * 3;

        // Extract RGB values and convert to grayscale
        float r = d_images[rgbPixelIdx] / 255.0f;
        float g = d_images[rgbPixelIdx + 1] / 255.0f;
        float b = d_images[rgbPixelIdx + 2] / 255.0f;

        // Using the luminosity method to convert to grayscale
        float gray = 0.21f * r + 0.72f * g + 0.07f * b;

        // Store the grayscale value in the output array
        d_images_float[imageIdx * numPixels + pixelIdx] = gray;

        // Convert labels to float (only once per image)
        if (pixelIdx == 0) {
            d_labels_float[imageIdx] = static_cast<float>(d_labels[imageIdx]);
        }
    }
}


void preprocessImage(unsigned char* d_images, float** d_images_float,
    unsigned char* d_labels, float** d_labels_float)
{
    int totalPixels = IMG_SIZE / 3 * NUM_IMAGES * DATA_BATCHES; // Total number of pixels in all images
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
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    printf("Preprocessing complete\n");
}