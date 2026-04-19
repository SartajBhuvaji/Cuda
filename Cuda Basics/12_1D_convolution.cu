// 1D Convolution
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<assert.h>

__global__ void convolution_1d(int* arr, int* mask, int* result, int n, int m) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int radius = m / 2; // calculate the radius of the mask

	int start = tid - radius;
	int temp = 0;

	for (int i = 0; i < m; i++) { // each thread calculates the convolution for one element
		if (start + i >= 0 && start + i < n) {
			temp += arr[start + i] * mask[i];
		}
	}
	result[tid] = temp;
}

void verify_result(int* arr, int* mask, int* output, int N, int M) {
	int radius = M / 2;
	int temp;
	int start;
	for (int i = 0; i < N; i++) {
		temp = 0;
		start = i - radius;
		for (int j = 0; j < M; j++) {
			if (start + j >= 0 && start + j < N) {
				temp += arr[start + j] * mask[j];
			}
		}
		assert(output[i] == temp);
	}
}

void fillArray(int* A, int N) {
	for (int i = 0; i < N; i++) {
		A[i] = rand() % 10;
	}
}

int main() {

	int N = 1 << 10; // 1024 element array
	int M = 3; // Mask size or the convolution window size (preferably an odd no)

	int size = N * sizeof(float);

	int* arr, * mask, * output;
	int* d_arr, * d_mask, * d_output;

	arr = (int*)malloc(size);
	mask = (int*)malloc(M * sizeof(int));
	output = (int*)malloc(size);

	cudaMalloc(&d_arr, size);
	cudaMalloc(&d_mask, M * sizeof(int));
	cudaMalloc(&d_output, size);

	fillArray(arr, N);
	fillArray(mask, M);

	cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, M * sizeof(int), cudaMemcpyHostToDevice);

	int THREADS = 256;
	int GRID = (N + THREADS - 1) / THREADS;

	convolution_1d << <GRID, THREADS >> > (d_arr, d_mask, d_output, N, M);

	cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

	verify_result(arr, mask, output, N, M);
	printf("Success\n");
}