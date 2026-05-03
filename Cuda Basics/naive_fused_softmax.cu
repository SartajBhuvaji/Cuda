#include <cuda.h>
#include <stdio.h>
#include <math.h>

__global__ void max_value(float* input, float* max_val, int size) {
    __shared__ float shared_max[256]; // Shared memory for block-level reduction
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    shared_max[tid] = -INFINITY;
    if (idx < size) {
        shared_max[tid] = input[idx];
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // Write the block's maximum value to global memory
    if (tid == 0) {
       float max_val = -INFINITY;
        for (int i = 0; i < size; i++) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }
    }
}

__global__ void softmax(float* input, float* output, float* max_val, int size) {
    __shared__ float shared_sum[256]; // Shared memory for block-level sum
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    // Load the maximum value from global memory
    float max_value = *max_val;

    // Compute the exponential values and store in shared memory
    shared_sum[tid] = 0.0f;
    if (idx < size) {
        shared_sum[tid] = expf(input[idx] - max_value);
    }
    __syncthreads();

    // Perform reduction in shared memory to compute the sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Compute the softmax output
    if (idx < size) {
        output[idx] = expf(input[idx] - max_value) / shared_sum[0];
    }
}

int main() {
    int size = 1 << 20; // 1M elements
    float* host_input = (float*)malloc(size * sizeof(float));
    float* host_output = (float*)malloc(size * sizeof(float));
    float host_max_val = -INFINITY;
    float* device_input;
    float* device_output;
    float* device_max_val;

    cudaEvent_t start, stop;

    // Initialize input with random values
    for (int i = 0; i < size; i++) {
        host_input[i] = rand() / (float)RAND_MAX; // Random values between 0 and 1
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&device_input, size * sizeof(float));
    cudaMalloc(&device_output, size * sizeof(float));
    cudaMalloc(&device_max_val, sizeof(float));

    cudaMemcpy(device_input, host_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_max_val, &host_max_val, sizeof(float), cudaMemcpyHostToDevice);

    int no_of_threads = 256;
    int no_of_blocks = (int)ceil((float)size / no_of_threads);

    cudaEventRecord(start);

    // Launch the kernel to compute the maximum value
    max_value<<<no_of_blocks, no_of_threads>>>(device_input, device_max_val, size);
    cudaDeviceSynchronize();

    // Launch the kernel to compute the softmax output
    softmax<<<no_of_blocks, no_of_threads>>>(device_input, device_output, device_max_val, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(host_output, device_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the elapsed time
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Print the first 10 output values
    for (int i = 0; i < 10; i++) {
        printf("Softmax Output[%d]: %f\n", i, host_output[i]);
    }

    free(host_input);
    free(host_output);
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_max_val);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();

    return 0;
}