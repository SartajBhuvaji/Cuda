#include<cuda.h>
#include<stdio.h>


__global__ void matAdd(int* A, int*B, int* C) {

	int i = threadIdx,x;
	int j = threadIDx.y;

	C[i][j] = A[i][j] + B[i][j];
}

int main() {

	int* A, * B, * C;	
	//fill in matrix A, B

	&A = { {1,2,3,4}, {5,6,7,8}, {9,10,11,12}, {13,14,15,16} };
	&B = { {1,2,3,4}, {5,6,7,8}, {9,10,11,12}, {13,14,15,16} };
}
