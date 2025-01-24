#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <curand_kernel.h>

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
        float grad_w = 0.0f;
        float grad_norm = 0.0f;
        
        for (int b = 0; b < batchSize; ++b) {
            float input_val = input[b * inputSize + row];
            float grad = gradients[b * outputSize + col];
            grad_w += input_val * grad;
            grad_norm += grad * grad;
        }
        
        // Normalize gradients
        grad_norm = sqrtf(grad_norm / batchSize + 1e-7f);
        grad_w = grad_w / (grad_norm * batchSize);
        
        atomicAdd(&grad_weights[row * outputSize + col], grad_w);

        if (row == 0) {
            float grad_b = 0.0f;
            for (int b = 0; b < batchSize; ++b) {
                grad_b += gradients[b * outputSize + col];
            }
            atomicAdd(&grad_biases[col], grad_b / (grad_norm * batchSize));
        }
    }
}

// Add gradient propagation kernel
__global__ void gradientPropagationKernel(float* input_gradients, float* output_gradients, 
                                         float* weights, float* input, 
                                         int inputSize, int outputSize, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    
    if (idx < inputSize && batch < batchSize) {
        float sum = 0.0f;
        for (int j = 0; j < outputSize; ++j) {
            float grad = output_gradients[batch * outputSize + j];
            grad = fmaxf(fminf(grad, 1.0f), -1.0f);  // Clip gradient
            sum += grad * weights[idx * outputSize + j];
        }
        
        // Leaky ReLU gradient
        float input_val = input[batch * inputSize + idx];
        float grad = sum * (input_val > 0.0f ? 1.0f : 0.01f);
        grad = fmaxf(fminf(grad, 1.0f), -1.0f);  // Clip gradient
        
        input_gradients[batch * inputSize + idx] = grad;
    }
}

__global__ void dropoutKernel(float* input, float* output, int size, float dropout_rate, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Initialize CURAND state
        curandState state;
        curand_init(seed, idx, 0, &state);  // Initialize state with a unique seed

        float random_val = curand_uniform(&state);  // Generate a random number in [0, 1)
        output[idx] = (random_val < dropout_rate) ? 0.0f : input[idx];
    }
}

__device__ unsigned int lcg_state = 0;  // Global state for LCG

__device__ float lcg_random() {
    lcg_state = (1103515245 * lcg_state + 12345) & 0x7fffffff;  // LCG formula
    return (float)lcg_state / (float)0x7fffffff;  // Normalize to [0, 1)
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
        // Xavier/Glorot initialization
        float scale = sqrtf(6.0f / (inputSize + outputSize));
        float* h_weights = new float[inputSize * outputSize];
        float* h_biases = new float[outputSize];

        // Initialize weights with Xavier/Glorot initialization
        for (int i = 0; i < inputSize * outputSize; ++i) {
            h_weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
        
        // Initialize biases to small values
        for (int i = 0; i < outputSize; ++i) {
            h_biases[i] = 0.01f;
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

        // Apply dropout
        unsigned long long seed = 1234;  // You can use a different seed for each call
        dropoutKernel<<<gridDim, blockDim>>>(d_output, d_output, outputSize * batchSize, 0.5f, seed); // 50% dropout

        return d_output;
    }


    int getOutputSize() const { return outputSize; }
    int getBatchSize() const { return batchSize; }
    float* getWeights() const { return d_weights; }
    float* getBiases() const { return d_biases; }
    float* getGradWeights() const { return d_grad_weights; }
    float* getGradBiases() const { return d_grad_biases; }

    void DenseLayer::backward(float* d_input, float* d_gradients) {
    // Reset gradients
    cudaMemset(d_grad_weights, 0, inputSize * outputSize * sizeof(float));
    cudaMemset(d_grad_biases, 0, outputSize * sizeof(float));
    cudaMemset(d_input_gradients, 0, inputSize * batchSize * sizeof(float));

    // Calculate gradients
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (outputSize + blockDim.x - 1) / blockDim.x,
        (inputSize + blockDim.y - 1) / blockDim.y
    );

    denseBackwardKernel<<<gridDim, blockDim>>>(d_input, d_gradients, d_grad_weights, d_grad_biases,
                                               inputSize, outputSize, batchSize);
    cudaDeviceSynchronize();

    // Propagate gradients to the previous layer
    dim3 propBlockDim(256);
    dim3 propGridDim(
        (inputSize + propBlockDim.x - 1) / propBlockDim.x,
        batchSize
    );

    gradientPropagationKernel<<<propGridDim, propBlockDim>>>(d_input_gradients, d_gradients, d_weights, d_input,
                                                              inputSize, outputSize, batchSize);
    cudaDeviceSynchronize();
}

    float* getInputGradients() const { return d_input_gradients; }

    float* getVelocityWeights() const { return d_velocity_weights; }
    float* getVelocityBiases() const { return d_velocity_biases; }
};