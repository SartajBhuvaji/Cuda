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


// CUDA kernel for convolution
//__global__ void convolutionKernel(float* input, float* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels, float* filter) {
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    int z = blockIdx.z; // for multiple images
//
//    __shared__ float sharedFilter[FILTER_SIZE * FILTER_SIZE];
//
//    // Load filter into shared memory
//    if (threadIdx.x < FILTER_SIZE && threadIdx.y < FILTER_SIZE) {
//        sharedFilter[threadIdx.y * FILTER_SIZE + threadIdx.x] = filter[threadIdx.y * FILTER_SIZE + threadIdx.x];
//    }
//    __syncthreads(); // Wait for all threads to load the filter
//
//    if (x < outputWidth && y < outputHeight) {
//        for (int c = 0; c < channels; ++c) {
//            float sum = 0.0f;
//            for (int fy = 0; fy < FILTER_SIZE; ++fy) {
//                for (int fx = 0; fx < FILTER_SIZE; ++fx) {
//                    int imgX = x + fx;
//                    int imgY = y + fy;
//
//                    int inputIndex = (z * inputHeight * inputWidth + imgY * inputWidth + imgX) * channels + c;
//                    sum += input[inputIndex] * sharedFilter[fy * FILTER_SIZE + fx];
//                }
//            }
//            int outIndex = (z * outputHeight * outputWidth + y * outputWidth + x) * channels + c;
//			output[outIndex] = fmaxf(sum, 0.0f); // ReLU activation // NEED TO FIX THIS
//        }
//    }
//}

__global__ void convolutionKernel(float* input, float* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels, float* filter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z; // for multiple images

    if (x < outputWidth && y < outputHeight) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int fy = 0; fy < FILTER_SIZE; ++fy) {
                for (int fx = 0; fx < FILTER_SIZE; ++fx) {
                    int imgX = x + fx;
                    int imgY = y + fy;

                    int inputIndex = (z * inputHeight * inputWidth + imgY * inputWidth + imgX) * channels + c;
                    int filterIndex = (fy * FILTER_SIZE + fx) * channels + c;
                    sum += input[inputIndex] * filter[filterIndex];

                    // Debugging: Print intermediate values
                    if (x == 0 && y == 0 && z == 0 && c == 0) {
                        printf("Conv: input[%d] = %f, filter[%d] = %f, product = %f\n",
                            inputIndex, input[inputIndex], filterIndex, filter[filterIndex],
                            input[inputIndex] * filter[filterIndex]);
                    }
                }
            }
            int outIndex = (z * outputHeight * outputWidth + y * outputWidth + x) * channels + c;
            output[outIndex] = sum;

            // Debugging: Print final sum
            if (x == 0 && y == 0 && z == 0 && c == 0) {
                printf("Conv: Final sum for output[%d] = %f\n", outIndex, sum);
            }
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

class ConvolutionLayer {
private:
    int inputWidth, inputHeight, inputChannels;
    int outputWidth, outputHeight, outputChannels;
    int poolOutputWidth, poolOutputHeight, poolOutputChannels;
    int batchSize;
    float* d_filters;  // Device memory for filters
    float* d_output;   // Device memory for output

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
    }

    ~ConvolutionLayer() {
        cudaFree(d_filters);
        cudaFree(d_output);
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
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "initializeFiltersKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
            outputWidth, outputHeight, inputChannels, d_filters);

        cudaDeviceSynchronize();

        //float h_output[10];
        //cudaMemcpy(h_output, d_output, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        //printf("Convolution output (first 10 values):\n");
        //for (int i = 0; i < 10; ++i) {
        //    printf("%f ", h_output[i]);
        //}
        //printf("\n");

        // Perform max pooling
        MaxPoolingLayer pool1(getOutputWidth(), getOutputHeight(), getOutputChannels(), batchSize);
        float* d_pool_output = pool1.forward(d_output);

        poolOutputWidth = pool1.getOutputWidth();
        poolOutputHeight = pool1.getOutputHeight();
        poolOutputChannels = pool1.getOutputChannels();

        // Perform ReLU activation
        float* d_activated_output = nullptr;
        cudaMalloc(&d_activated_output, poolOutputWidth * poolOutputHeight * poolOutputChannels * batchSize * sizeof(float));
        applyActivation(d_pool_output, d_activated_output, poolOutputWidth * poolOutputHeight * poolOutputChannels * batchSize, "relu");

        return d_activated_output;
    }

    void updateFilters(float* gradients, float learningRate) {
        // This is a placeholder for the actual update logic
        // you'd apply the gradients to update the filters
        
        // cudaMemcpy(h_filters, d_filters, filterSize, cudaMemcpyDeviceToHost);
        // Update h_filters using gradients and learning rate
        // cudaMemcpy(d_filters, h_filters, filterSize, cudaMemcpyHostToDevice);
    }

	void backprop(float* gradients) {
		// This is a placeholder for the actual backpropagation logic
		//  you'd calculate the gradients and backpropagate them
		
        //updateFilters();
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