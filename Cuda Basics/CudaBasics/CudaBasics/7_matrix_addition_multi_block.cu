#include <cuda.h>
#include <stdio.h>
#define N 4

__global__ void matAdd(int* A, int* B, int* C) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    // Ensure that indices are within the matrix bounds
    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

int main() {
    int size = N * N * sizeof(int);

    // Allocate host memory
    int h_A[N * N], h_B[N * N], h_C[N * N]; // Matrix A, B, C of size N x N

    // Initialize matrices on the host
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = i;
        h_B[i] = i;
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
    dim3 threadsPerBlock(N, N); // We create a 2D grid of N x N threads
    int numBlocks = 1;

    // Launch the kernel
    matAdd << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result matrix
    printf("Result matrix C:\n");
    for (int i = 0; i < N * N; ++i) {
        printf("%d ", h_C[i]);
        if ((i + 1) % N == 0) {
            printf("\n");
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
