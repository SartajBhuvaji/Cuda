#include <float.h>
#include <math.h>
#include <stdio.h>

__global__ void softmax_shared(float* input, float* output, int size) {
    extern __shared__ float sdata[]; // shared memory 


    // In a shared-memory reduction kernel, we often don’t use the global idx
    // for computation inside the kernel logic, because the kernel is 
    // designed so that each block works on its own chunk of data independently.
    int tid = threadIdx.x;
    int idx = tid;

    // 1. Load into shared memory
    float val = (idx < size) ? input[idx] : -FLT_MAX;
    // We purposefully fill out-of-bounds because we are using tiling
    // and we want to make sure that the reduction works correctly
    sdata[tid] = val;
    __syncthreads();

    // 2. REDUCE MAX (parallel reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)  {
        // stride >>= 1 is equivalent to stride /= 2, but it's more efficient (somehow)
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    // Thus using stride we do sdata[0] = max(x0, x4), sdata[1] = max(x1, x5) and so on until we get the final max in sdata[0]
    // After the reduction, sdata[0] contains the maximum value for this block
    // Thus using multiple threads we can compute the max value in O(log(blockDim.x)) 
    // time instead of O(blockDim.x) time if we were to do it sequentially


    float max_val = sdata[0];  // final max

    // 3. Compute exp(x - max)
    float exp_val = (idx < size) ? expf(input[idx] - max_val) : 0.0f;
    sdata[tid] = exp_val;
    __syncthreads();

    // 4. REDUCE SUM
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float sum_val = sdata[0];  // final sum

    // 5. Normalize
    if (idx < size) {
        output[idx] = exp_val / sum_val;
    }
}

int main() {
    int size = 1 << 10; // 1024
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
    // Launch the kernel with shared memory size equal to blockDim.x * sizeof(float)
    softmax_shared<<<no_of_blocks, no_of_threads, no_of_threads * sizeof(float)>>>(device_input, device_output, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time taken: %f ms\n", milliseconds);

    // Copy back the results and free memory
    cudaMemcpy(host_output, device_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    free(host_input);
    free(host_output);
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}
