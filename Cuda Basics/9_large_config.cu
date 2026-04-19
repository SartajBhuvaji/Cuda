// Launch Configuration for Huge Data
// Launching kernel for huge data
/*
We have vector of huge size and we want to launch thread for each element of array.

*/


#include <cuda.h>
#include <stdio.h>

#define BLOCKSIZE 1024 // block cannot have size > 1024

__global__ void dkernel(unsigned* vector, unsigned vectorsize) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < vectorsize) vector[id] = id;
}

int main() {
	unsigned N = 1025; // denotes the no of threads and the vecor size that user wants to launch
	unsigned* vector, * h_vector;

	cudaMalloc(&vector, N * sizeof(unsigned));
	h_vector = (unsigned*)malloc(N * sizeof(unsigned));

	unsigned nblocks = ceil((float)N / BLOCKSIZE); // calculating no of blocks
	printf("Number of blocks: %d\n", nblocks);

	dkernel << <nblocks, BLOCKSIZE >> > (vector, N);
	cudaMemcpy(h_vector, vector, N * sizeof(unsigned), cudaMemcpyDeviceToHost);

	for (unsigned i = 0; i < 100; i++) {
		printf("%d ", h_vector[i]);
	}
	return 0;
}