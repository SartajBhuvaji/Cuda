#include<cuda.h>
#include<stdio.h>


__global__ void sum(int* A, int stride) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i % (2 * stride) == 0) {
		A[i] = A[i] + A[i + stride];
	}
}

int main() {

	int A[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
	int* dA;
	cudaMalloc(&dA, 16 * sizeof(int));
	cudaMemcpy(dA, A, 16 * sizeof(int), cudaMemcpyHostToDevice);

	for (int stride = 1; stride < 16; stride *= 2) {
		sum << <1, 16 >> > (dA, stride);
	}

	cudaMemcpy(A, dA, 16 * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 16; i++) {
		printf("%d ", A[i]);
	}
	printf("\n");

	cudaFree(dA);
	return 0;

}

/*
Given an array of length 16

Step 1:
i is even  A[i] = A[i] + A[i+1]

Step 2
i is multiple of 4  A[i] = A[i] + A[i+2]

Step 3
i is multiple of 8  A[i] = A[i] + A[i+4]

Step 4
i is multiple of 16  A[i] = A[i] + A[i+8]

*/

/*
* Output:
120 2 7 4 26 6 15 8 84 10 23 12 42 14 15 0
*/

// Reference : https://www.youtube.com/watch?v=8U1vRxIUz5A
