#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "max_pooling.cu"
#include "activations.cu"

#define FILTER_SIZE 3

// Existing convolutionKernel remains the same...

// Kernel to compute gradients for convolution filters
__global__ void convBackwardKernel(float* d_output, float* d_gradients, float* d_input, float* d_grad_filters, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels, int batchSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z; // batch index

    if (x < outputWidth && y < outputHeight && b < batchSize) {
        for (int c = 0; c < channels; ++c) {
            for (int fy = 0; fy < FILTER_SIZE; ++fy) {
                for (int fx = 0; fx < FILTER_SIZE; ++fx) {
                    int imgX = x + fx;
                    int imgY = y + fy;

                    int inputIndex = ((b * inputHeight * inputWidth + imgY * inputWidth + imgX) * channels) + c;
                    int filterIndex = (fy * FILTER_SIZE + fx) * channels + c;
                    float grad = d_input[inputIndex] * d_gradients[((b * outputHeight * outputWidth + y * outputWidth + x) * channels) + c];
                    atomicAdd(&d_grad_filters[filterIndex], grad);
                }
            }
        }
    }
}

class ConvolutionLayer {
private:
    int inputWidth, inputHeight, inputChannels;
    int outputWidth, outputHeight, outputChannels;
    int poolOutputWidth, poolOutputHeight, poolOutputChannels;
    int batchSize;
    float* d_filters;          // Device memory for filters
    float* d_output;           // Device memory for output
    float* d_grad_filters;     // Device memory for filter gradients

public:
    ConvolutionLayer(int inWidth, int inHeight, int inChannels, int batchSz)
        : inputWidth(inWidth), inputHeight(inHeight), inputChannels(inChannels), batchSize(batchSz) {
        outputWidth = inputWidth - FILTER_SIZE + 1;
        outputHeight = inputHeight - FILTER_SIZE + 1;
        outputChannels = inputChannels; // Simple case: same number of output channels as input

        // Allocate and initialize filters
        cudaMalloc(&d_filters, FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels * sizeof(float));
        initializeFilters();

        // Allocate memory for output and filter gradients
        cudaMalloc(&d_output, outputWidth * outputHeight * outputChannels * batchSize * sizeof(float));
        cudaMalloc(&d_grad_filters, FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels * sizeof(float)); // Initialize to zero
        cudaMemset(d_grad_filters, 0, FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels * sizeof(float));
    }

    ~ConvolutionLayer() {
        cudaFree(d_filters);
        cudaFree(d_output);
        cudaFree(d_grad_filters);
    }

    void initializeFilters() {
        int total_elements = FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels;
        float* h_filters = new float[total_elements];
        float scale = sqrt(2.0f / (FILTER_SIZE * FILTER_SIZE * inputChannels + FILTER_SIZE * FILTER_SIZE * outputChannels));

        for (int i = 0; i < total_elements; ++i) {
            h_filters[i] = scale * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        }

        cudaMemcpy(d_filters, h_filters, total_elements * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_filters;
    }

    float* forward(float* d_input) {
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (outputWidth + blockDim.x - 1) / blockDim.x,
            (outputHeight + blockDim.y - 1) / blockDim.y,
            batchSize
        );

        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, inputWidth, inputHeight,
            outputWidth, outputHeight, inputChannels, d_filters, batchSize);

        cudaDeviceSynchronize();

        // Perform max pooling and activation
        MaxPoolingLayer pool1(getOutputWidth(), getOutputHeight(), getOutputChannels(), batchSize);
        float* d_pool_output = pool1.forward(d_output);

        poolOutputWidth = pool1.getOutputWidth();
        poolOutputHeight = pool1.getOutputHeight();
        poolOutputChannels = pool1.getOutputChannels();

        float* d_activated_output = nullptr;
        cudaMalloc(&d_activated_output, poolOutputWidth * poolOutputHeight * poolOutputChannels * batchSize * sizeof(float));
        applyActivation(d_pool_output, d_activated_output, poolOutputWidth * poolOutputHeight * poolOutputChannels * batchSize, "relu");

        return d_activated_output;
    }

    void backward(float* d_input, float* d_gradients) {
        // Reset gradients to zero
        cudaMemset(d_grad_filters, 0, FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels * sizeof(float));

        dim3 blockDim(16, 16);
        dim3 gridDim(
            (outputWidth + blockDim.x - 1) / blockDim.x,
            (outputHeight + blockDim.y - 1) / blockDim.y,
            batchSize
        );

        convBackwardKernel<<<gridDim, blockDim>>>(d_output, d_gradients, d_input, d_grad_filters, inputWidth, inputHeight, outputWidth, outputHeight, inputChannels, batchSize);

        cudaDeviceSynchronize();
    }

    // Getters for filter gradients
    float* getGradFilters() const { return d_grad_filters; }

    // Getters for filters
    float* getFilters() const { return d_filters; }

    // Getters for output dimensions
    int getOutputWidth() const { return outputWidth; }
    int getOutputHeight() const { return outputHeight; }
    int getOutputChannels() const { return outputChannels; }

    // Getters for pool output dimensions
    int getPoolOutputWidth() const { return poolOutputWidth; }
    int getPoolOutputHeight() const { return poolOutputHeight; }
    int getPoolOutputChannels() const { return poolOutputChannels; }
}; 