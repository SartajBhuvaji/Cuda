#include<cuda.h>
#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void simple_loop(int n) {   // __global__ is a GPU kernel function specifier
	for (int i = 0; i < n; i++) {
		printf("% d ", i * i);
	}
}

__global__ void threaded_loop() {
	int index = threadIdx.x; // threadIdx is a built-in variable provided by CUDA at runtime that gives the index of the thread in the block
	printf("%d ", index * index);
}

int main() {
	int n = 10;
	//simple_loop << <1, 1 >> > (n);  // <<<1,1>>> is a kernel launch configuration // 1 block and 1 thread
	threaded_loop << <1, n >> > ();  // <<<1,1>>> is a kernel launch configuration // 1 block and n threads

	cudaDeviceSynchronize();   // cudaDeviceSynchronize() is a function that waits for the device to finish its execution
	return 0;
}
t i = 0; i < n; i++) {
		printf(i*i);
	}
}

int main() {
	int n = 10;
	dkernel << <1, 1 >> > (n);  // <<<1,1>>> is a kernel launch configuration // 1 block and 1 thread
	cudaDeviceSynchronize();   // cudaDeviceSynchronize() is a function that waits for the device to finish its execution
	return 0;
}
