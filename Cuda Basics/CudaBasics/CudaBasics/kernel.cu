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
	printf("%d %d %d = %d\n", threadIdx.x, blockDim.y, threadIdx.y, id);
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
			printf("%d ", hmatrix[ii * M + jj]);
		}
		printf("\n");
	}

	cudaDeviceSynchronize();
	return 0;
}


// Output:
/*
* Actual representation of the matrix in the 1D array in the GPU's global memory
* | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 |


* Output of the program
| 0 | 1 | 2 | 3 | 4 | 5  |
| 6 | 7 | 8 | 9 | 10 | 11 |
| 12 | 13 | 14 | 15 | 16 | 17 |
| 18 | 19 | 20 | 21 | 22 | 23 |
| 24 | 25 | 26 | 27 | 28 | 29 |

*/
/*

*Explanation:

 In CUDA, when you represent a matrix on the GPU, it is typically stored as a 1D array in the GPU's global memory. 
 This is because CUDA does not natively support multidimensional arrays in global memory. 
 Instead, you use a flat 1D array and manually calculate the indices to access elements as if they were in a 2D (or higher-dimensional) structure.

*/

/*

Mapping a 2D Matrix to a 1D Array
To represent a 2D matrix in a 1D array, you typically map the 2D coordinates to a 1D index using the following formula:

1D index = row index * number of columns + column index

*/
