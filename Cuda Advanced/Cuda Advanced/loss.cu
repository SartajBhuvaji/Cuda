#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Kernel for cross-entropy loss calculation
__global__ void crossEntropyLossKernel(float* predictions, float* targets, 
                                      float* loss, int numClasses, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        float sample_loss = 0.0f;
        for (int c = 0; c < numClasses; ++c) {
            float pred = fmaxf(fminf(predictions[idx * numClasses + c], 1.0f - 1e-7f), 1e-7f);
            sample_loss -= targets[idx * numClasses + c] * logf(pred);
        }
        atomicAdd(loss, sample_loss);
    }
}

float calculateLoss(float* predictions, float* targets, int numClasses, int batchSize) {
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    dim3 block(256);
    dim3 grid((batchSize + block.x - 1) / block.x);
    
    crossEntropyLossKernel<<<grid, block>>>(predictions, targets, d_loss, numClasses, batchSize);
    
    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    
    return h_loss / batchSize;
} 