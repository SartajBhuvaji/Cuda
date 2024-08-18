#include<cuda.h>
#include<stdio.h>
#include<math.h>

__global__ void vectorAdd(int* A, int* B, int* C, int N) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

void matrixFill(int* A, int N) {
	for (int i = 0; i < N; i++) {
		A[i] = rand() % 100;
	}
}

int main() {
	// Step 1 : Initialize the variables
	int n = 1 << 16;

	// Step 2 : Allocate memory on the host
	int* h_a, * h_b, * h_c;
	// Step 3 : Allocate memory on the device
	int* d_a, * d_b, * d_c;

	// Step 4 : Calculate the size of the memory
	size_t bytes = n * sizeof(int);

	// Step 5 : Allocate memory on the host
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	// Step 6 : Allocate memory on the device
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Step 7 : Fill the matrix
	matrixFill(h_a, n);
	matrixFill(h_b, n);

	// Step 8 : Copy the data from host to device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// Step 9 : Setup the execution configuration
	int num_threads = 256;
	int num_blocks = (int)ceil(n / num_threads);

	// Step 10 : Execute the kernel
	vectorAdd << <num_blocks, num_threads >> > (d_a, d_b, d_c, n);

	// Step 11 : Copy the result back to the host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Step 12 : Display the result
	for (int i = 0; i < 10; i++) {
		printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
	}

	// Step 13 : Free the memory on the host
	free(h_a);
	free(h_b);
	free(h_c);

	// Step 14 : Free the memory on the device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
