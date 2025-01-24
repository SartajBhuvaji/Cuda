#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

// Helper function to check CUDA errors
inline void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

// Print GPU memory usage information
inline void gpu_mem_info() {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << "\nGPU memory usage: used = " << used_db / 1024.0 / 1024.0 
              << "MB, free = " << free_db / 1024.0 / 1024.0 
              << "MB, total = " << total_db / 1024.0 / 1024.0 << "MB" << std::endl;
}

// Gradient clipping kernel
__global__ void clipGradientsKernel(float* gradients, int size, float max_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (gradients[idx] > max_value) gradients[idx] = max_value;
        if (gradients[idx] < -max_value) gradients[idx] = -max_value;
    }
}

// Wrapper function for gradient clipping
inline void clipGradients(float* d_gradients, int size, float max_value) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    clipGradientsKernel<<<grid, block>>>(d_gradients, size, max_value);
    cudaDeviceSynchronize();
}

// Input normalization kernel
__global__ void normalizeInputsKernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = input[idx] / 255.0f;  // Normalize to [0,1]
    }
}

#endif // CUDA_UTILS_CUH 