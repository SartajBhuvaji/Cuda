#ifndef ACTIVATIONS_CUH
#define ACTIVATIONS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Function declarations
__global__ void leakyReluKernel(float* input, float* output, int size, float alpha = 0.01f);
__global__ void sigmoidKernel(float* input, float* output, int size);
__global__ void tanhKernel(float* input, float* output, int size);
__global__ void eluKernel(float* input, float* output, int size, float alpha = 1.0f);
__global__ void seluKernel(float* input, float* output, int size);
__global__ void reluKernel(float* input, float* output, int size);
__global__ void softmaxKernel(float* input, float* output, int batchSize, int numClasses);
__global__ void reluActivationKernel(float* input, float* output, int size);
__global__ void reluBackwardKernel(float* input, float* gradOutput, float* gradInput, int size);

void applyActivation(float* input, float* output, int size, const char* activationType, int classes = 10);

#endif // ACTIVATIONS_CUH 