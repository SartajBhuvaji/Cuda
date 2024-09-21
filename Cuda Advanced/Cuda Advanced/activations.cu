// activations.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>



// Leaky ReLU activation function
__global__ void leakyReluKernel(float* input, float* output, int size, float alpha = 0.01f) {
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
__global__ void eluKernel(float* input, float* output, int size, float alpha = 1.0f) {
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
//__global__ void softmaxKernel(float* input, float* output, int size, int classes) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < size) {
//        int start = (idx / classes) * classes;
//        int end = start + classes;
//
//        float max_val = input[start];
//        for (int i = start + 1; i < end; ++i) {
//            max_val = fmaxf(max_val, input[i]);
//        }
//
//        float sum = 0.0f;
//        for (int i = start; i < end; ++i) {
//            sum += expf(input[i] - max_val);
//        }
//
//        output[idx] = expf(input[idx] - max_val) / sum;
//    }
//}

// ReLU (Rectified Linear Unit) activation function
__global__ void reluKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);

        // Debugging
        if (idx < 10) {
            printf("ReLU: Input[%d] = %f, Output[%d] = %f\n", idx, input[idx], idx, output[idx]);
        }
    }
}

// Softmax activation function (for the last layer of classification networks)
__global__ void softmaxKernel(float* input, float* output, int size, int classes) {
    int batchIdx = blockIdx.x;
    int classIdx = threadIdx.x;

    if (classIdx < classes) {
        int start = batchIdx * classes;
        int end = start + classes;

        // Find max value for numerical stability
        float max_val = input[start];
        for (int i = start + 1; i < end; ++i) {
            max_val = fmaxf(max_val, input[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        float exp_vals[64];  // Assuming max 64 classes, adjust if needed

        for (int i = 0; i < classes; ++i) {
            exp_vals[i] = expf(input[start + i] - max_val);
            sum += exp_vals[i];
        }

        // Normalize
        output[start + classIdx] = exp_vals[classIdx] / sum;

        // Debugging
        if (batchIdx == 0 && classIdx < 10) {
            printf("Softmax: Input[%d] = %f, Output[%d] = %f\n",
                start + classIdx, input[start + classIdx],
                start + classIdx, output[start + classIdx]);
        }
    }
}



// Wrapper function to launch activation kernels
void applyActivation(float* input, float* output, int size, const char* activationType, int classes = 10) {
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

    cudaDeviceSynchronize();
}