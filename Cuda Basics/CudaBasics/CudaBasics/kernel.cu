// Grid, Block, Thread

/*
Thread Organization
- A grid has blocks and a block has threads
- A kernel is executed as a grid of thread blocks
- A grid is a 3D array of thread blocks (gridDim.x, gridDim.y, gridDim.z)
- Each thread block has a unique index within the grid (blockIdx.x, blockIdx.y, blockIdx.z)
- A thread block is a 3D array of threads (blockDim.x, blockDim.y, blockDim.z)
- Each thread has a unique index within the block (threadIdx.x, threadIdx.y, threadIdx.z)


Typical Configuration
- 1-5 blocks per SM(Streaming Multiprocessor)
- Each SM is assigned one or more blocks
- 128-1024 threads per block
- Total 2k-100k threads
- You can launch a kernel with millions of threads
*/


#include<cuda.h>
#include<stdio.h>



__global__ void dkernel() {
	if (threadIdx.x == 0 && blockIdx.x == 0 &&
		threadIdx.y == 0 && blockIdx.y == 0 &&
		threadIdx.z == 0 && blockIdx.z == 0
		) {
		printf("%d %d %d %d %d %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
	}
}


#define N 5
#define M 6
__global__ void dkernel_2D(unsigned* matrix) {
	unsigned id = threadIdx.x * blockDim.y + threadIdx.y;
	matrix[id] = id;
}

int main() {

	// For : dkernel
	//dim3 block(2, 3, 4); // 3D block of threads
	//dim3 thread(5, 6, 7); // 
	//dkernel << <block, thread >> > ();
	
	// For:dkernel_2D
	dim3 block(N, M, 1); // 2D block of threads of 5*6 = 30 threads
	unsigned* matrix, * hmatrix;

	cudaMalloc(&matrix, N * M * sizeof(unsigned));
	hmatrix = (unsigned*)malloc(N * M * sizeof(unsigned));

	dkernel_2D << <1, block >> > (matrix);

	cudaMemcpy(hmatrix, matrix, N * M * sizeof(unsigned), cudaMemcpyDeviceToHost);

	for (unsigned ii = 0; ii < N; ++ii) {
		for (unsigned jj = 0; jj < M; ++jj) {
			printf("%2d ", hmatrix[ii * M + jj]);
		}
		printf("\n");
	}

	cudaDeviceSynchronize();
	return 0;
}