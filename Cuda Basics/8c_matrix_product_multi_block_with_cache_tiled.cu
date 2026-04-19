//Cache Tiled Matrix Multiplication with Shared Memory(L1 Cache)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

using namespace std;

// defining shared memory size// static allcocation
#define SHMEM_SIZE (16 * 16) // 16x16 tile size

__global__ void matrixMul(int* a, int* b, int* c, int N) {
	// Shared memory can be allocated dynamically at runtime or preallocated at compile time
	//Allocating shared memory
	__shared__ int A[SHMEM_SIZE]; // shared memory is private to each block
	__shared__ int B[SHMEM_SIZE];

	// Calculate global thread indices
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Extract some built-in variables
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int dim = blockDim.x;
	// Move the tile across the length of the grid
	int temp = 0;
	for (int i = 0; i < ((N + dim - 1) / dim); ++i) {
		A[ty * dim + tx] = a[row * N + i * dim + tx];
		B[ty * dim + tx] = b[(i * dim + ty) * N + col];
		__syncthreads();


		// Accumulate the partial results
		for (int j = 0; j < dim; ++j) {
			if (i * dim + j < N) {
				temp += A[ty * dim + j] * B[j * dim + tx];
			}
			__syncthreads();
		}
		c[row * N + col] = temp;
	}
}


void verify_result(int* a, int* b, int* c, int N) {
	int tmp;
	for (int i = 0; i < N; i++) {
		for (int j = 0; i < N; j++) {
			tmp = 0;
			for (int k = 0; i < N; k++) {
				tmp = a[i * N + k] * b[k * N + j];
			}
			assert(tmp == c[i * N + j]);
		}
	}

}


void matrixInit(int* a, int N) {
	for (int i = 0; i < N * N; i++) {
		a[i] = rand() % 100;
	}
}

int main() {
	// Set problem size
	int N = 1 << 10; //2^10
	size_t size = N * N * sizeof(int);

	//Allocate memory for the matrices
	int* a, * b, * c;
	int* d_a, * d_b, * d_c;

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	// Initialize matrix
	matrixInit(a, N);
	matrixInit(b, N);

	// Copy matrices to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// setup cuda config
	int thread = 16;
	int  blocks = (N + thread - 1) / thread;

	dim3 THREADS(thread, thread);
	dim3 BLOCKS(blocks, blocks);

	matrixMul << < BLOCKS, THREADS >> > (a, b, c, N);
	cudaDeviceSynchronize();

	// verify the result on CPU
	verify_result(a, b, c, N);
	cout << "Completed Successfully" << endl;

	return 0;
}
