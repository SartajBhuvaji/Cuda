#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void matMul(int* A, int* B, int* C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n && j < n) {
		int sum = 0;
		for (int k = 0; k < n; ++k) {
			sum += A[i * n + k] * B[k * n + j];
		}
		C[i * n + j] = sum;
	}
}

void fillMatrix(int* matrix, int n) {
	for (int i = 0; i < n * n; ++i) {
		matrix[i] = rand() % 100;
	}
}

int main() {
	int n = 1 << 16;

	int* h_a, * h_b, * h_c;
	int* d_a, * d_b, * d_c;

	size_t bytes = n * n * sizeof(int);

	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	fillMatrix(h_a, n);
	fillMatrix(h_b, n);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	int noOfThreads = 1024;
	int noOfBlocks = (int)ceil(n / noOfThreads);

	matMul << <noOfBlocks, noOfThreads >> > (d_a, d_b, d_c, n);

	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Print result matrix
	printf("Result matrix C:\n");
	for (int i = 0; i < 10; ++i) {
		printf("%d ", h_c[i]);
		if ((i + 1) % n == 0) {
			printf("\n");
		}
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);

	return 0;

}