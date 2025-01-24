#ifndef ACTIVATION_FUNCTIONS_CUH
#define ACTIVATION_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

// Activation functions ot the quasu! 
__device__ float relu(float x) { return fmaxf(0.0f, x); }
__device__ float leaky_relu(float x, float alpha = 0.01f) { return fmaxf(alpha * x, x); }
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ float tanh_activation(float x) { return tanhf(x); }
__device__ float elu(float x, float alpha = 1.0f) { return x >= 0 ? x : alpha * (expf(x) - 1); }
__device__ float selu(float x) {
    const float alpha = 1.6733f;
    const float scale = 1.0507f;
    return scale * (x >= 0 ? x : alpha * (expf(x) - 1));
}
__device__ float softplus(float x) { return logf(1 + expf(x)); }
__device__ float swish(float x) { return x * sigmoid(x); }
__device__ float gelu(float x) { return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / CUDART_PI_F) * (x + 0.044715f * powf(x, 3)))); }
__device__ float mish(float x) { return x * tanhf(softplus(x)); }

// Enum for activation types
enum ActivationType {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    ELU,
    SELU,
    SOFTPLUS,
    SWISH,
    GELU,
    MISH,
    IDENTITY
};

// Apply activation function
static __device__ float apply_activation(float x, ActivationType activation_type) {
    switch (activation_type) {
    case RELU: return relu(x);
    case LEAKY_RELU: return leaky_relu(x);
    case SIGMOID: return sigmoid(x);
    case TANH: return tanh_activation(x);
    case ELU: return elu(x);
    case SELU: return selu(x);
    case SOFTPLUS: return softplus(x);
    case SWISH: return swish(x);
    case GELU: return gelu(x);
    case MISH: return mish(x);
    default: return x;  // IDENTITY
    }
}

// Kernel to apply activation function to a vector
__global__ void apply_activation_kernel(float* input, float* output, int size, ActivationType activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = apply_activation(input[idx], activation_type);
    }
}

#endif // ACTIVATION_FUNCTIONS_CUH




// Usage

#include "activation_functions.cuh"
#include <vector>

int main() {
    std::vector<float> host_input = { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f };
    int vector_size = host_input.size();
	ActivationType activation_type = RELU; // Change this to test different activation functions
     
    // Allocate device memory
    float* d_input, * d_output;
    cudaMalloc(&d_input, vector_size * sizeof(float));
    cudaMalloc(&d_output, vector_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, host_input.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);

    // Set up kernel launch parameters
    int blockSize = 256;
    int gridSize = (vector_size + blockSize - 1) / blockSize;

    // Launch kernel
    apply_activation_kernel << <gridSize, blockSize >> > (d_input, d_output, vector_size, activation_type);

    // Copy result back to host
    std::vector<float> host_output(vector_size);
    cudaMemcpy(host_output.data(), d_output, vector_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    // Print or process results
    for (float val : host_output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}