#include <cuda.h>
#include <stdio.h>

#define N 4  // Size of matrix (N x N)

__global__ void matAdd(int* A, int* B, int* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate row index in multi-block grid
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Calculate column index in multi-block grid

    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

int main() {
    int size = N * N * sizeof(int); // Size of matrices in bytes

    // Allocate host memory
    int h_A[N * N], h_B[N * N], h_C[N * N]; // Matrix A, B, C of size N x N but stored as 1D arrays 

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
    dim3 threadsPerBlock(N, N);  // We create a 2D grid of N x N threads
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y); // We calculate the number of blocks needed based on the matrix size

    matAdd << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result matrix
    printf("Result matrix C:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
