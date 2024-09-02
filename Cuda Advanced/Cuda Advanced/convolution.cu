﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cmath>

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5   // Total number of data batches
#define FILTER_SIZE 3

// Struct to hold convolution results
struct ConvolutionResult {
    float* output;
    float* kernel;
    int outputWidth;
    int outputHeight;
    int outputChannels;
};

// Updated to return a 1D array for easier usage with CUDA
float* initialize_kernel(int n, const std::string& initializer) {
    std::srand(static_cast<unsigned>(std::time(0)));
    float scale;
    if (initializer == "Xavier") {
        scale = std::sqrt(2.0f / (n * n)); // Xavier initialization
    }
    else if (initializer == "He") {
        scale = std::sqrt(2.0f / (n * n)); // He initialization
    }
    else {
        std::cerr << "Unknown initializer " << initializer << std::endl;
        return nullptr;
    }

    float* kernel = new float[n * n];

    for (int i = 0; i < n * n; ++i) {
        kernel[i] = scale * (static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    printf("Kernel initialized\n");
    return kernel;
}


// Doc https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_64_website/projects/convolutionSeparable/doc/convolutionSeparable.pdf
__global__ void convolutionKernel(float* input, float* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels, float* filter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z; // for multiple images

    __shared__ float sharedFilter[FILTER_SIZE * FILTER_SIZE];

    // Load filter into shared memory
    if (threadIdx.x < FILTER_SIZE && threadIdx.y < FILTER_SIZE) {
		sharedFilter[threadIdx.y * FILTER_SIZE + threadIdx.x] = filter[threadIdx.y * FILTER_SIZE + threadIdx.x]; 
    }
    __syncthreads(); // Wait for all threads to load the filter

    if (x < outputWidth && y < outputHeight) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int fy = 0; fy < FILTER_SIZE; ++fy) {
                for (int fx = 0; fx < FILTER_SIZE; ++fx) {
                    int imgX = x + fx;
                    int imgY = y + fy; 

                    int inputIndex = (z * inputHeight * inputWidth + imgY * inputWidth + imgX) * channels + c; 
                    sum += input[inputIndex] * sharedFilter[fy * FILTER_SIZE + fx];
                }
            }
            int outIndex = (z * outputHeight * outputWidth + y * outputWidth + x) * channels + c;
            output[outIndex] = sum;
        }
    }
}

// Broken
//void validateConvolutionOutput(const float* input, int inputWidth, int inputHeight, int inputChannels,
//    const float* output, int filterSize, int stride = 1, int padding = 0) {
//    // Calculate expected output dimensions
//	printf("\nIN VALIDATE CONVOLUTION OUTPUT\n");
//    int outputWidth = static_cast<int>(std::floor((inputWidth + 2 * padding - filterSize) / stride + 1));
//    int outputHeight = static_cast<int>(std::floor((inputHeight + 2 * padding - filterSize) / stride + 1));
//	int outputChannels = inputChannels;  // Number of channels remains the same (3)
//
//    // Calculate expected total elements in the output
//    int expectedOutputSize = outputWidth * outputHeight * outputChannels;
//
//    // Calculate actual output size
//    int actualOutputSize = 0;
//    while (output[actualOutputSize] != 0 && actualOutputSize < inputWidth * inputHeight * inputChannels) {
//        actualOutputSize++;
//    }
//
//    // Assert that the calculated size matches the actual size
//    assert(expectedOutputSize ==  && "Convolution output size mismatch");
//
//    // If assertion passes, print success message
//    printf("Convolution output size validation passed.\n");
//    printf("Input dimensions: %d x %d x %d\n", inputWidth, inputHeight, inputChannels);
//    printf("Filter size: %d x %d\n", filterSize, filterSize);
//    printf("Output dimensions: %d x %d x %d\n", outputWidth, outputHeight, outputChannels);
//  //  printf("Total output elements: %d\n", actualOutputSizeexpectedOutputSize); //?
//}

ConvolutionResult convolution(float* d_images_float, float* d_labels_float, int inputWidth, int inputHeight, int numImages) {
    printf("IN CONVOLUTION\n");
    int channels = 3;
    int outputWidth = inputWidth - FILTER_SIZE + 1;
    int outputHeight = inputHeight - FILTER_SIZE + 1;

    // Allocate device memory for output
    float* d_output;
    cudaMalloc(&d_output, outputWidth * outputHeight * channels * numImages * sizeof(float));

    // Create and initialize the filter using the initialize_kernel function
    float* h_filter = initialize_kernel(FILTER_SIZE, "Xavier"); // or "He"

    // Allocate device memory for filter and copy it to device
    float* d_filter;
    cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim(
        (outputWidth + blockDim.x - 1) / blockDim.x,
        (outputHeight + blockDim.y - 1) / blockDim.y,
        numImages
    );

    convolutionKernel << <gridDim, blockDim >> > (d_images_float, d_output, inputWidth, inputHeight, outputWidth, outputHeight, channels, d_filter);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    printf("Convolution done\n");

    // Allocate host memory for output and copy result from device to host
    float* h_output = (float*)malloc(outputWidth * outputHeight * channels * numImages * sizeof(float));
    cudaMemcpy(h_output, d_output, outputWidth * outputHeight * channels * numImages * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first few elements of the output (for debugging)
    printf("First few elements of convolution output:\n");
    int counter = 0;
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            std::cout << h_output[j + i * IMG_SIZE] << " ";
            counter++;
        }
        std::cout << std::endl;
    }
	printf("Total number of pixels: %d\n", counter);
    printf("Output dimensions: %d x %d x %d\n", outputWidth, outputHeight, channels);

    // Free device memory
    cudaFree(d_output);

    // Prepare and return the result
    ConvolutionResult result;
    result.output = h_output;
    result.kernel = h_filter;
    result.outputWidth = outputWidth;
    result.outputHeight = outputHeight;
    result.outputChannels = channels;

    return result;
}