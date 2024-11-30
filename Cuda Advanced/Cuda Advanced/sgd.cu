#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Kernel for SGD update
__global__ void sgdWeightsKernel(float* d_weights, float* d_gradients, int size, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Clip gradients to prevent explosion
        float gradient = d_gradients[idx];
        float max_grad = 1.0f;
        if (gradient > max_grad) gradient = max_grad;
        if (gradient < -max_grad) gradient = -max_grad;
        
        d_weights[idx] -= learningRate * gradient;
    }
}

void sgdUpdateWeights(float* d_weights, float* d_grad_weights, int size, float learningRate) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sgdWeightsKernel<<<numBlocks, blockSize>>>(d_weights, d_grad_weights, size, learningRate);
    cudaDeviceSynchronize();
}

__global__ void sgdBiasesKernel(float* d_biases, float* d_grad_biases, int size, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_biases[idx] -= learningRate * d_grad_biases[idx];
    }
}

void sgdUpdateBiases(float* d_biases, float* d_grad_biases, int size, float learningRate) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sgdBiasesKernel<<<numBlocks, blockSize>>>(d_biases, d_grad_biases, size, learningRate);
    cudaDeviceSynchronize();
} 