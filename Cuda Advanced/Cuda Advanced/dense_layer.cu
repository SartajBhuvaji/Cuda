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

class DenseLayer {
private:
    int inputSize, outputSize, batchSize;
    float* d_weights;  // Device memory for weights
    float* d_biases;   // Device memory for biases
    float* d_output;   // Device memory for output

public:
    DenseLayer(int inSize, int outSize, int batchSize)
        : inputSize(inSize), outputSize(outSize), batchSize(batchSize) {
        // Allocate memory for weights and biases
        cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_biases, outputSize * sizeof(float));
        cudaMalloc(&d_output, outputSize * batchSize * sizeof(float));

        // Initialize weights and biases
        initializeParameters();
    }

    ~DenseLayer() {
        cudaFree(d_weights);
        cudaFree(d_biases);
        cudaFree(d_output);
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
            h_biases[i] = scale * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
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

    // Placeholder for backpropagation function
    void backpropagate(const float* gradients, float learningRate) {
        // TODO: Implement backpropagation
        // This will update weights and biases based on the gradients
    }

    int getOutputSize() const { return outputSize; }
    int getBatchSize() const { return batchSize; }
    float* getWeights() const { return d_weights; }
    float* getBias() const { return d_biases; }
};