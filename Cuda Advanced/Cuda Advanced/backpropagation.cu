#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Kernel for backpropagation
__global__ void backpropKernel(float* d_output, float* d_target, float* d_gradients, 
                              float* d_intermediate_gradients,
                              int outputSize, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = outputSize * batchSize;
    
    if (idx < total_elements) {
        int batch_idx = idx / outputSize;
        int class_idx = idx % outputSize;
        
        // Cross-entropy gradient: (predicted - actual)
        float predicted = d_output[idx];
        float target = d_target[batch_idx * outputSize + class_idx];
        
        // Compute gradient
        float gradient = predicted - target;
        
        // Store the gradient
        d_gradients[idx] = gradient;
        
        // Also compute intermediate gradients for previous layers if needed
        if (d_intermediate_gradients != nullptr) {
            // Derivative of softmax * cross entropy
            d_intermediate_gradients[idx] = gradient * predicted * (1.0f - predicted);
        }
    }
}

// Main backpropagation function
void backpropagate(float* d_output, float* d_target, float* d_gradients, 
                  float* d_intermediate_gradients,
                  int outputSize, int batchSize) {
    // Error checking
    if (d_output == nullptr || d_target == nullptr || d_gradients == nullptr) {
        fprintf(stderr, "Error: Null pointer passed to backpropagate\n");
        return;
    }

    int blockSize = 256;
    int numBlocks = (outputSize * batchSize + blockSize - 1) / blockSize;
    
    backpropKernel<<<numBlocks, blockSize>>>(
        d_output, 
        d_target, 
        d_gradients,
        d_intermediate_gradients,
        outputSize, 
        batchSize
    );
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to launch backprop kernel: %s\n", cudaGetErrorString(error));
        return;
    }
    
    cudaDeviceSynchronize();
} 