// activations.cu

#include "activations.cuh"
#include <cmath>

// Leaky ReLU activation function
__global__ void leakyReluKernel(float* input, float* output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : alpha * input[idx];
    }
}

// Sigmoid activation function
__global__ void sigmoidKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Tanh activation function
__global__ void tanhKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// ELU (Exponential Linear Unit) activation function
__global__ void eluKernel(float* input, float* output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] >= 0 ? input[idx] : alpha * (expf(input[idx]) - 1);
    }
}

// SELU (Scaled Exponential Linear Unit) activation function
__global__ void seluKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float alpha = 1.6732632423543772848170429916717f;
        const float scale = 1.0507009873554804934193349852946f;
        float x = input[idx];
        output[idx] = scale * (x >= 0 ? x : alpha * (expf(x) - 1));
    }
}

// Softmax activation function (for the last layer of classification networks)
__global__ void softmaxKernel(float* input, float* output, int batchSize, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        // Find max value for numerical stability
        float maxVal = -INFINITY;
        for (int i = 0; i < numClasses; ++i) {
            maxVal = fmaxf(maxVal, input[idx * numClasses + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < numClasses; ++i) {
            float val = expf(input[idx * numClasses + i] - maxVal);
            output[idx * numClasses + i] = val;
            sum += val;
        }

        // Normalize
        for (int i = 0; i < numClasses; ++i) {
            output[idx * numClasses + i] /= sum;
        }
    }
}

// ReLU (Rectified Linear Unit) activation function
__global__ void reluKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

__global__ void reluActivationKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(0.0f, input[idx]);
    }
}

__global__ void reluBackwardKernel(float* input, float* gradOutput, float* gradInput, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradInput[idx] = input[idx] > 0.0f ? gradOutput[idx] : 0.0f;
    }
}

// Wrapper function to launch activation kernels
void applyActivation(float* input, float* output, int size, const char* activationType, int classes) {
    cudaError_t error;
   
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    if (strcmp(activationType, "relu") == 0) {
        reluKernel << <numBlocks, blockSize >> > (input, output, size);
    }
    else if (strcmp(activationType, "leaky_relu") == 0) {
        leakyReluKernel << <numBlocks, blockSize >> > (input, output, size);
    }
    else if (strcmp(activationType, "sigmoid") == 0) {
        sigmoidKernel << <numBlocks, blockSize >> > (input, output, size);
    }
    else if (strcmp(activationType, "tanh") == 0) {
        tanhKernel << <numBlocks, blockSize >> > (input, output, size);
    }
    else if (strcmp(activationType, "elu") == 0) {
        eluKernel << <numBlocks, blockSize >> > (input, output, size);
    }
    else if (strcmp(activationType, "selu") == 0) {
        seluKernel << <numBlocks, blockSize >> > (input, output, size);
    }
    else if (strcmp(activationType, "softmax") == 0) {
        softmaxKernel << <numBlocks, blockSize >> > (input, output, size, classes);
    }
    else {
        printf("Unknown activation function: %s\n", activationType);
    }

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in activation: %s\n", cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();
}