#include <cuda.h>
#include <stdio.h>

#define N 10000  // Size of matrix (N x N)

__global__ void matMul(int* A, int* B, int* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

int main() {
    int size = N * N * sizeof(int);

    // Allocate host memory
    int* h_A = new int[N * N];
    int* h_B = new int[N * N];
    int* h_C = new int[N * N];

    // Initialize matrices on the host
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = i % 100; // Example initialization
        h_B[i] = i % 100;
    }

    // Allocate device memory
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16); // Example block size
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matMul << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C,);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Optionally, print a portion of the result matrix if desired
    printf("Result matrix C (part):\n");
    for (int i = 0; i < N && i < 10; ++i) {
        for (int j = 0; j < N && j < 10; ++j) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
