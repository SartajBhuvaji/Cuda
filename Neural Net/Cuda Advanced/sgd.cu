#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Kernel for SGD update
void sgdUpdateWeights(float* d_weights, float* d_gradients, float* d_velocity,
                      int size, float learningRate, float momentum, float weightDecay) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sgdWeightsKernel<<<numBlocks, blockSize>>>(d_weights, d_gradients, d_velocity,
                                              size, learningRate, momentum, weightDecay);
    cudaDeviceSynchronize();
}

__global__ void sgdWeightsKernel(float* d_weights, float* d_gradients, float* d_velocity,
                                 int size, float learningRate, float momentum, float weightDecay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply weight decay
        d_gradients[idx] += weightDecay * d_weights[idx];
        // Update velocity
        d_velocity[idx] = momentum * d_velocity[idx] - learningRate * d_gradients[idx];
        // Update weights
        d_weights[idx] += d_velocity[idx];
    }
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