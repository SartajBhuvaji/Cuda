#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Kernel for backpropagation
__global__ void backpropKernel(float* d_output, float* d_target, float* d_gradients, int outputSize, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = outputSize * batchSize;
    
    if (idx < total_elements) {
        int batch_idx = idx / outputSize;
        int class_idx = idx % outputSize;
        
        // Cross-entropy gradient: (predicted - actual)
        float predicted = d_output[idx];
        float target = d_target[batch_idx * outputSize + class_idx];
        d_gradients[idx] = (predicted - target);
    }
}

void backpropagate(float* d_output, float* d_target, float* d_gradients, int outputSize, int batchSize) {
    int blockSize = 256;
    int numBlocks = (outputSize * batchSize + blockSize - 1) / blockSize;
    backpropKernel<<<numBlocks, blockSize>>>(d_output, d_target, d_gradients, outputSize, batchSize);
    cudaDeviceSynchronize();
} 