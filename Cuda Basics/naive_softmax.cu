#include<cuda.h>
#include<stdio.h>
#include<math.h>

__global__ void softmax(float* input, float* output, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Compute the maximum value
    // This is not efficient as all threads compute the max value, but it's a naive implementation
    if (idx < size) {
        float max_val = -INFINITY;
        for (int i = 0; i < size; i++) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }

        // Compute the sum of exponentials
        float sum_exp = 0.0f;
        for (int i = 0; i < size; i++) {
            sum_exp += expf(input[i] - max_val);
        }

        // Compute the softmax output
        output[idx] = expf(input[idx] - max_val) / sum_exp;
    }
}

int main() {
    int size = 1 << 10;
    float* host_input = (float*)malloc(size * sizeof(float));
    float* host_output = (float*)malloc(size * sizeof(float));
    float* device_input;
    float* device_output;
    cudaEvent_t start, stop;

    // Initialize input with random values
    for (int i = 0; i < size; i++) {
        host_input[i] = rand() / (float)RAND_MAX; // Random values between 0 and 1
    }

    cudaMalloc(&device_input, size * sizeof(float));
    cudaMalloc(&device_output, size * sizeof(float));

    cudaMemcpy(device_input, host_input, size * sizeof(float), cudaMemcpyHostToDevice);

    int no_of_threads = 256;
    int no_of_blocks = (int)ceil((float)size / no_of_threads);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Launch the kernel
    softmax<<<no_of_blocks, no_of_threads>>>(device_input, device_output, size);
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();

    return 0;
}
