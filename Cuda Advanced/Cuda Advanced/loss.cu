#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Kernel for cross-entropy loss calculation
__global__ void crossEntropyLossKernel(float* d_output, float* d_target, float* d_loss, int outputSize, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        float loss = 0.0f;
        for (int i = 0; i < outputSize; ++i) {
            float target = d_target[idx * outputSize + i];
            float output = d_output[idx * outputSize + i];
            loss -= target * logf(output + 1e-9); // Add epsilon to prevent log(0)
        }
        d_loss[idx] = loss;
    }
}

float calculateLoss(float* d_output, float* d_target, int outputSize, int batchSize) {
    float* d_loss;
    cudaMalloc(&d_loss, batchSize * sizeof(float));

    int blockSize = 256;
    int numBlocks = (batchSize + blockSize - 1) / blockSize;
    crossEntropyLossKernel<<<numBlocks, blockSize>>>(d_output, d_target, d_loss, outputSize, batchSize);
    cudaDeviceSynchronize();

    float* h_loss = new float[batchSize];
    cudaMemcpy(h_loss, d_loss, batchSize * sizeof(float), cudaMemcpyDeviceToHost);

    float totalLoss = 0.0f;
    for (int i = 0; i < batchSize; ++i) {
        totalLoss += h_loss[i];
    }

    delete[] h_loss;
    cudaFree(d_loss);

    return totalLoss / batchSize;
} 