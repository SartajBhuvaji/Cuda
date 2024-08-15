#include<cuda.h>
#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void mem_copy(int* a) {   // __global__ is a GPU kernel function specifier
	a[threadIdx.x] = threadIdx.x * threadIdx.x;
}

int main() {
	const int n = 10;
	int a[n], * d_a;

	cudaMalloc(&d_a, n * sizeof(int));  // Allocate memory on the device, d_a is the pointer to the memory on the device
	mem_copy << <1, n >> > (d_a);

	cudaDeviceSynchronize();
	cudaMemcpy(a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);  // Copy data from device to host
	cudaFree(d_a); // Free the memory on the GPU

	for (int i = 0; i < n; i++) {
		printf("%d ", a[i]);
	}

	return 0;
}
