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

// Kernel for dense layer backward pass
__global__ void denseBackwardKernel(float* input, float* gradients, float* grad_weights, float* grad_biases, 
                                   int inputSize, int outputSize, int batchSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < inputSize && col < outputSize) {
        // Calculate gradient for weights
        float grad_w = 0.0f;
        for (int b = 0; b < batchSize; ++b) {
            grad_w += input[b * inputSize + row] * gradients[b * outputSize + col];
        }
        grad_weights[row * outputSize + col] = grad_w;

        // Calculate gradient for biases (only one thread per column)
        if (row == 0) {
            float grad_b = 0.0f;
            for (int b = 0; b < batchSize; ++b) {
                grad_b += gradients[b * outputSize + col];
            }
            grad_biases[col] = grad_b;
        }
    }
}

// Add gradient propagation kernel
__global__ void gradientPropagationKernel(float* input_gradients, float* output_gradients, 
                                         float* weights, int inputSize, int outputSize, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    
    if (idx < inputSize && batch < batchSize) {
        float sum = 0.0f;
        for (int j = 0; j < outputSize; ++j) {
            sum += output_gradients[batch * outputSize + j] * weights[idx * outputSize + j];
        }
        input_gradients[batch * inputSize + idx] = sum;
    }
}

class DenseLayer {
private:
    int inputSize, outputSize, batchSize;
    float* d_weights;  // Device memory for weights
    float* d_biases;   // Device memory for biases
    float* d_output;   // Device memory for output
    float* d_grad_weights;  // Device memory for weight gradients
    float* d_grad_biases;   // Device memory for bias gradients
    float* d_velocity_weights;  // Add momentum
    float* d_velocity_biases;
    const float momentum = 0.9f;
    float* d_input_gradients;  // Add this for storing input gradients

public:
    DenseLayer(int inSize, int outSize, int batchSize)
        : inputSize(inSize), outputSize(outSize), batchSize(batchSize) {
        // Allocate memory for weights and biases
        cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_biases, outputSize * sizeof(float));
        cudaMalloc(&d_output, outputSize * batchSize * sizeof(float));
        cudaMalloc(&d_grad_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_grad_biases, outputSize * sizeof(float));
        cudaMalloc(&d_velocity_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_velocity_biases, outputSize * sizeof(float));
        cudaMemset(d_velocity_weights, 0, inputSize * outputSize * sizeof(float));
        cudaMemset(d_velocity_biases, 0, outputSize * sizeof(float));
        cudaMalloc(&d_input_gradients, inputSize * batchSize * sizeof(float));

        // Initialize weights and biases
        initializeParameters();
    }

    ~DenseLayer() {
        cudaFree(d_weights);
        cudaFree(d_biases);
        cudaFree(d_output);
        cudaFree(d_grad_weights);
        cudaFree(d_grad_biases);
        cudaFree(d_velocity_weights);
        cudaFree(d_velocity_biases);
        cudaFree(d_input_gradients);
    }

    void initializeParameters() {
        // Initialize weights using He initialization
        float scale = sqrt(2.0f / inputSize);  // He initialization
        float* h_weights = new float[inputSize * outputSize];
        float* h_biases = new float[outputSize];

        for (int i = 0; i < inputSize * outputSize; ++i) {
            h_weights[i] = scale * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        }
        
        // Initialize biases to small values
        for (int i = 0; i < outputSize; ++i) {
            h_biases[i] = 0.01f * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
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


    int getOutputSize() const { return outputSize; }
    int getBatchSize() const { return batchSize; }
    float* getWeights() const { return d_weights; }
    float* getBiases() const { return d_biases; }
    float* getGradWeights() const { return d_grad_weights; }
    float* getGradBiases() const { return d_grad_biases; }

    void backward(float* d_input, float* d_gradients) {
        // Reset gradients
        cudaMemset(d_grad_weights, 0, inputSize * outputSize * sizeof(float));
        cudaMemset(d_grad_biases, 0, outputSize * sizeof(float));
        cudaMemset(d_input_gradients, 0, inputSize * batchSize * sizeof(float));

        // Calculate gradients for weights and biases
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (outputSize + blockDim.x - 1) / blockDim.x,
            (inputSize + blockDim.y - 1) / blockDim.y
        );

        denseBackwardKernel<<<gridDim, blockDim>>>(
            d_input, d_gradients, d_grad_weights, d_grad_biases,
            inputSize, outputSize, batchSize
        );

        // Propagate gradients to previous layer
        dim3 propBlockDim(256);
        dim3 propGridDim(
            (inputSize + propBlockDim.x - 1) / propBlockDim.x,
            batchSize
        );

        gradientPropagationKernel<<<propGridDim, propBlockDim>>>(
            d_input_gradients, d_gradients, d_weights,
            inputSize, outputSize, batchSize
        );

        cudaDeviceSynchronize();
    }

    float* getInputGradients() const { return d_input_gradients; }
    float* getVelocityWeights() const { return d_velocity_weights; }
    float* getVelocityBiases() const { return d_velocity_biases; }
};