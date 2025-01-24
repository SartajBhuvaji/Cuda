#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

#define POOL_SIZE 2
#define POOL_STRIDE 2

__global__ void maxPoolingKernel(float* input, float* output, int inputWidth, int inputHeight, int channels, int outputWidth, int outputHeight, int numImages) {
    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z; // for multiple images

    if (outputX < outputWidth && outputY < outputHeight) {
        for (int c = 0; c < channels; ++c) {
            float maxVal = -INFINITY;
            for (int py = 0; py < POOL_SIZE; ++py) {
                for (int px = 0; px < POOL_SIZE; ++px) {
                    int inputX = outputX * POOL_STRIDE + px;
                    int inputY = outputY * POOL_STRIDE + py;
                    if (inputX < inputWidth && inputY < inputHeight) {
                        int inputIndex = (z * inputHeight * inputWidth + inputY * inputWidth + inputX) * channels + c;
                        maxVal = fmaxf(maxVal, input[inputIndex]);
                    }
                }
            }
            int outIndex = (z * outputHeight * outputWidth + outputY * outputWidth + outputX) * channels + c;
            output[outIndex] = maxVal;
        }
    }
}

class MaxPoolingLayer {
private:
    int inputWidth, inputHeight, channels, numImages;
    int outputWidth, outputHeight;
    float* d_output;

public:
    MaxPoolingLayer(int inWidth, int inHeight, int inChannels, int inNumImages)
        : inputWidth(inWidth), inputHeight(inHeight), channels(inChannels), numImages(inNumImages) {
        outputWidth = inputWidth / POOL_STRIDE;
        outputHeight = inputHeight / POOL_STRIDE;

        cudaMalloc(&d_output, outputWidth * outputHeight * channels * numImages * sizeof(float));
    }

    ~MaxPoolingLayer() {
        cudaFree(d_output);
    }

    float* forward(float* d_input) {
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (outputWidth + blockDim.x - 1) / blockDim.x,
            (outputHeight + blockDim.y - 1) / blockDim.y,
            numImages
        );

        maxPoolingKernel << <gridDim, blockDim >> > (d_input, d_output, inputWidth, inputHeight,
            channels, outputWidth, outputHeight, numImages);

        cudaDeviceSynchronize();
        return d_output;
    }

    // Getter methods
    int getOutputWidth() const { return outputWidth; }
    int getOutputHeight() const { return outputHeight; }
    int getOutputChannels() const { return channels; }
};