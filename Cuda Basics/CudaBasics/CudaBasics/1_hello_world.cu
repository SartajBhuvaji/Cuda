#include<cuda.h>
#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void dkernel() {   // __global__ is a GPU kernel function specifier
	printf("Hello World");
}

int main() {
	dkernel << <1, 1 >> > ();  // <<<1,1>>> is a kernel launch configuration // 1 block and 1 thread
	cudaDeviceSynchronize();   // cudaDeviceSynchronize() is a function that waits for the device to finish its execution
	return 0;
}
