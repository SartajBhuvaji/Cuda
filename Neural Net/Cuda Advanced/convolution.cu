#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\max_pooling.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\activations.cu>

#define FILTER_SIZE 3

// Kernel for computing gradients during backpropagation
__global__ void convBackwardKernel(float* d_output, float* d_gradients, float* d_input, float* d_grad_filters, 
    int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels, int batchSize) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z; // batch index

    if (x < outputWidth && y < outputHeight && b < batchSize) {
        for (int c = 0; c < channels; ++c) {
            for (int fy = 0; fy < FILTER_SIZE; ++fy) {
                for (int fx = 0; fx < FILTER_SIZE; ++fx) {
                    int imgX = x + fx;
                    int imgY = y + fy;
                    if (imgX < inputWidth && imgY < inputHeight) {
                        int inputIdx = ((b * inputHeight * inputWidth + imgY * inputWidth + imgX) * channels) + c;
                        int gradIdx = ((b * outputHeight * outputWidth + y * outputWidth + x) * channels) + c;
                        int filterIdx = (fy * FILTER_SIZE + fx) * channels + c;
                        
                        atomicAdd(&d_grad_filters[filterIdx], 
                                d_input[inputIdx] * d_gradients[gradIdx]);
                    }
                }
            }
        }
    }
}

__global__ void convolutionKernel(float* input, float* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels, float* filter, int batchSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z; // batch index

    if (x < outputWidth && y < outputHeight && b < batchSize) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int fy = 0; fy < FILTER_SIZE; ++fy) {
                for (int fx = 0; fx < FILTER_SIZE; ++fx) {
                    int imgX = x + fx;
                    int imgY = y + fy;

                    int inputIndex = ((b * inputHeight * inputWidth + imgY * inputWidth + imgX) * channels) + c;
                    int filterIndex = (fy * FILTER_SIZE + fx) * channels + c;
                    sum += input[inputIndex] * filter[filterIndex];
                }
            }
            int outIndex = ((b * outputHeight * outputWidth + y * outputWidth + x) * channels) + c;
            output[outIndex] = sum;
        }
    }
}


__global__ void initializeFiltersKernel(float* filters, int inputChannels, int outputChannels, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels;

    if (idx < total_elements) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        float fan_in = FILTER_SIZE * FILTER_SIZE * inputChannels;
        float fan_out = FILTER_SIZE * FILTER_SIZE * outputChannels;
        float limit = sqrt(6.0f / (fan_in + fan_out));

        filters[idx] = curand_uniform(&state) * 2.0f * limit - limit;
    }
}

// Add this kernel for filter updates
__global__ void updateFiltersKernel(float* filters, float* gradients, float learningRate, float weightDecay, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        // Apply weight decay and learning rate
        float update = learningRate * (gradients[idx] + weightDecay * filters[idx]);
        filters[idx] -= update;
    }
}

class ConvolutionLayer {
private:
    int inputWidth, inputHeight, inputChannels;
    int outputWidth, outputHeight, outputChannels;
    int poolOutputWidth, poolOutputHeight, poolOutputChannels;
    int batchSize;
    float* d_filters;  // Device memory for filters
    float* d_output;   // Device memory for output
    float* d_grad_filters;  // Device memory for filter gradients

public:
    ConvolutionLayer(int inWidth, int inHeight, int inChannels, int batchSize)
        : inputWidth(inWidth), inputHeight(inHeight), inputChannels(inChannels), batchSize(batchSize) {
        outputWidth = inputWidth - FILTER_SIZE + 1;
        outputHeight = inputHeight - FILTER_SIZE + 1;
        outputChannels = inputChannels; // Preserve the number of channels

        // Allocate and initialize filters
        cudaMalloc(&d_filters, FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels * sizeof(float));
        initializeFilters();

        // Allocate memory for output
        cudaMalloc(&d_output, outputWidth * outputHeight * outputChannels * batchSize * sizeof(float));

        // Allocate memory for filter gradients
        cudaMalloc(&d_grad_filters, FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels * sizeof(float));
    }

    ~ConvolutionLayer() {
        cudaFree(d_filters);
        cudaFree(d_output);
        cudaFree(d_grad_filters);
    }


    void initializeFilters() {
        outputChannels = inputChannels; 
        int total_elements = FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels;

        // Allocate device memory for filters
        cudaMalloc(&d_filters, total_elements * sizeof(float));
        int blockSize = 256;
        int gridSize = (total_elements + blockSize - 1) / blockSize;

        // Initialize random seed
        unsigned long long seed = 1234ULL;  // You can change this seed or make it random
        initializeFiltersKernel << <gridSize, blockSize >> > (d_filters, inputChannels, outputChannels, seed);

        // Check CUDA ERRORS
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "initialize Filters Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        cudaDeviceSynchronize();
    }


    //void initializeFilters() {
    //    float* h_filters = new float[FILTER_SIZE * FILTER_SIZE * inputChannels];
    //    // Xavier initialization
    //    float scale = sqrt(2.0f / (FILTER_SIZE * FILTER_SIZE * inputChannels));
    //    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE * inputChannels; ++i) {
    //        h_filters[i] = scale * (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    //    }
    //    cudaMemcpy(d_filters, h_filters, FILTER_SIZE * FILTER_SIZE * inputChannels * sizeof(float), cudaMemcpyHostToDevice);
    //    delete[] h_filters;
    //}


    float* forward(float* d_input) {
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (outputWidth + blockDim.x - 1) / blockDim.x,
            (outputHeight + blockDim.y - 1) / blockDim.y,
            batchSize
        );

        convolutionKernel << <gridDim, blockDim >> > (d_input, d_output, inputWidth, inputHeight,
            outputWidth, outputHeight, inputChannels, d_filters, batchSize);

        cudaDeviceSynchronize();

        // Perform max pooling and activation
        MaxPoolingLayer pool1(getOutputWidth(), getOutputHeight(), getOutputChannels(), batchSize);
        float* d_pool_output = pool1.forward(d_output);

        poolOutputWidth = pool1.getOutputWidth();
        poolOutputHeight = pool1.getOutputHeight();
        poolOutputChannels = pool1.getOutputChannels();

        float* d_activated_output = nullptr;
        cudaMalloc(&d_activated_output, poolOutputWidth * poolOutputHeight * poolOutputChannels * batchSize * sizeof(float));
        applyActivation(d_pool_output, d_activated_output, poolOutputWidth * poolOutputHeight * poolOutputChannels * batchSize, "relu");

        return d_activated_output;
    }

    void updateFilters(float* gradients, float learningRate) {
        int totalElements = FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels;
        
        // Launch kernel to update filters
        int blockSize = 256;
        int gridSize = (totalElements + blockSize - 1) / blockSize;
        
        float weightDecay = 0.0001f; // Add weight decay to prevent overfitting
        updateFiltersKernel<<<gridSize, blockSize>>>(d_filters, d_grad_filters, learningRate, weightDecay, totalElements);
        
        // Synchronize to ensure updates are complete
        cudaDeviceSynchronize();
        
        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "Failed to update filters: %s\n", cudaGetErrorString(error));
        }
    }

	void backward(float* d_input, float* d_gradients) {
        // Reset gradients to zero
        cudaMemset(d_grad_filters, 0, FILTER_SIZE * FILTER_SIZE * inputChannels * outputChannels * sizeof(float));

        dim3 blockDim(16, 16);
        dim3 gridDim(
            (outputWidth + blockDim.x - 1) / blockDim.x,
            (outputHeight + blockDim.y - 1) / blockDim.y,
            batchSize
        );

        // First compute gradients for filters
        convBackwardKernel<<<gridDim, blockDim>>>(
            d_output, 
            d_gradients, 
            d_input, 
            d_grad_filters,
            inputWidth, inputHeight, 
            outputWidth, outputHeight, 
            inputChannels, 
            batchSize
        );
        
        cudaDeviceSynchronize();
        
        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "Failed to compute filter gradients: %s\n", cudaGetErrorString(error));
            return;
        }

        // Update the filters using computed gradients
        updateFilters(d_grad_filters, 0.001f);
	}

    // Getter methods
    int getOutputWidth() const { return outputWidth; }
    int getOutputHeight() const { return outputHeight; }
    int getOutputChannels() const { return outputChannels; }

	int getPoolOutputWidth() const { return poolOutputWidth; }
	int getPoolOutputHeight() const { return poolOutputHeight; }
	int getPoolOutputChannels() const { return poolOutputChannels; }

	float* getFilters() const { return d_filters; }
    int getBatchSize() const { return batchSize; }

    float* getGradFilters() const { return d_grad_filters; }

};


//// Usage example
//int main() {
//    // Assume we have d_input allocated and filled with input data
//    float* d_input;
//    int inputWidth = 32, inputHeight = 32, inputChannels = 3;
//    int numImages = 64;
//
//    // Create a convolution layer
//    ConvolutionLayer conv1(inputWidth, inputHeight, inputChannels, numImages);
//
//    // Perform forward pass
//    float* d_output = conv1.forward(d_input);
//
//    // ... rest of the network ...
//
//    // During backpropagation
//    float* gradients;  // Assume this is calculated
//    float learningRate = 0.01;
//    conv1.updateFilters(gradients, learningRate);
//
//    return 0;
//}