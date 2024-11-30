#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Kernel for backpropagation
__global__ void backpropKernel(float* d_output, float* d_target, float* d_gradients, int outputSize, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize * batchSize) {
        d_gradients[idx] = (d_output[idx] - d_target[idx]) / batchSize;
    }
}

void backpropagate(float* d_output, float* d_target, float* d_gradients, int outputSize, int batchSize) {
    int blockSize = 256;
    int numBlocks = (outputSize * batchSize + blockSize - 1) / blockSize;
    backpropKernel<<<numBlocks, blockSize>>>(d_output, d_target, d_gradients, outputSize, batchSize);
    cudaDeviceSynchronize();
} 