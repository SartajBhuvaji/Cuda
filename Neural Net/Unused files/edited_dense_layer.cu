#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

// Kernel for dense layer forward pass
__global__ void denseForwardKernel(float* input, float* weights, float* biases, float* output, int inputSize, int outputSize, int batchSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batchSize && col < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; ++i) {
            sum += input[row * inputSize + i] * weights[i * outputSize + col];
        }
        output[row * outputSize + col] = sum + biases[col];
    }
}

// Kernel for calculating gradients of weights and biases
__global__ void denseBackwardKernel(float* d_input, float* d_gradients, float* d_grad_weights, float* d_grad_biases, int inputSize, int outputSize, int batchSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output neuron

    if (row < batchSize && col < outputSize) {
        // Compute gradient for biases
        atomicAdd(&d_grad_biases[col], d_gradients[row * outputSize + col]);

        // Compute gradient for weights
        for (int i = 0; i < inputSize; ++i) {
            float grad = d_input[row * inputSize + i] * d_gradients[row * outputSize + col];
            atomicAdd(&d_grad_weights[i * outputSize + col], grad);
        }
    }
}

class DenseLayer {
private:
    int inputSize, outputSize, batchSize;
    float* d_weights;       // Device memory for weights
    float* d_biases;        // Device memory for biases
    float* d_output;        // Device memory for output
    float* d_grad_weights;  // Device memory for weight gradients
    float* d_grad_biases;   // Device memory for bias gradients

public:
    DenseLayer(int inSize, int outSize, int batchSz)
        : inputSize(inSize), outputSize(outSize), batchSize(batchSz) {
        // Allocate memory for weights, biases, outputs, and gradients
        cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_biases, outputSize * sizeof(float));
        cudaMalloc(&d_output, outputSize * batchSize * sizeof(float));
        cudaMalloc(&d_grad_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_grad_biases, outputSize * sizeof(float));

        // Initialize weights and biases
        initializeParameters();
    }

    ~DenseLayer() {
        cudaFree(d_weights);
        cudaFree(d_biases);
        cudaFree(d_output);
        cudaFree(d_grad_weights);
        cudaFree(d_grad_biases);
    }

    void initializeParameters() {
        // Initialize weights using Xavier initialization
        float scale = sqrt(2.0f / (inputSize + outputSize));
        float* h_weights = new float[inputSize * outputSize];
        float* h_biases = new float[outputSize];

        for (int i = 0; i < inputSize * outputSize; ++i) {
            h_weights[i] = scale * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        }
        for (int i = 0; i < outputSize; ++i) {
            h_biases[i] = 0.0f; // Initialize biases to zero
        }

        cudaMemcpy(d_weights, h_weights, inputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases, h_biases, outputSize * sizeof(float), cudaMemcpyHostToDevice);

        delete[] h_weights;
        delete[] h_biases;
    }

    float* forward(float* d_input) {
        // Perform matrix multiplication and add biases
        dim3 blockDim(16, 16);
        dim3 gridDim((outputSize + blockDim.x - 1) / blockDim.x, (batchSize + blockDim.y - 1) / blockDim.y);

        // Launch a kernel to perform the forward pass
        denseForwardKernel<<<gridDim, blockDim>>>(d_input, d_weights, d_biases, d_output, inputSize, outputSize, batchSize);

        cudaDeviceSynchronize();
        return d_output;
    }

    void backward(float* d_input, float* d_gradients) {
        // Reset gradients to zero
        cudaMemset(d_grad_weights, 0, inputSize * outputSize * sizeof(float));
        cudaMemset(d_grad_biases, 0, outputSize * sizeof(float));

        dim3 blockDim(16, 16);
        dim3 gridDim((outputSize + blockDim.x - 1) / blockDim.x, (batchSize + blockDim.y - 1) / blockDim.y);

        // Launch a kernel to compute gradients
        denseBackwardKernel<<<gridDim, blockDim>>>(d_input, d_gradients, d_grad_weights, d_grad_biases, inputSize, outputSize, batchSize);

        cudaDeviceSynchronize();
    }

    // Getters for gradients
    float* getGradWeights() const { return d_grad_weights; }
    float* getGradBiases() const { return d_grad_biases; }

    // Getters for weights and biases
    float* getWeights() const { return d_weights; }
    float* getBiases() const { return d_biases; }
}; 