#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <cmath>
#include <cstdlib>
#include <ctime>

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5   // Total number of data batches


// TODO: Debug this kernel

__global__ void convolutionKernel_old(float* d_images, float* d_output, int width, int height, int outputWidth, int outputHeight, float** kernel) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	float sum = 0.0f;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			int imageX = x + i;
			int imageY = y + j;
			sum += d_images[imageY * width + imageX] * kernel[j][i];
		}
	}

	d_output[y * outputWidth + x] = sum;
}


float** initialize_kernel(int n, const std::string& initializer) {
	
	std::srand(static_cast<unsigned>(std::time(0)));
	float scale;
	if (initializer == "Xavier") {
		scale = std::sqrt(2.0f / (n + n)); // Xavier initialization
	}
	else if (initializer == "He") {
		scale = std::sqrt(2.0f / n);      // He initialization
	}
	else {
		std::cerr << "Unknown initializer: " << initializer << std::endl;
		return nullptr;
	}

	float** kernel = new float* [n];
	for (int i = 0; i < n; ++i) kernel[i] = new float[n];

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			kernel[i][j] = scale * (static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
		}
	}
	printf("Kernel initialized\n");
	return kernel;

}

void convolution(float* d_images_float, float* d_labels_float, int width, int height, int numImages) {
	int n = 3; 
	std::string initializer = "Xavier"; // Change to "Xavier" or "He"

	int outputWidth = width - 2;
	int outputHeight = height - 2;

	float* d_output;
	cudaMalloc(&d_output, outputWidth * outputHeight * numImages * sizeof(float));

	dim3 gridDim(numImages);
	dim3 blockDim(outputWidth, outputHeight);

	float** kernel = initialize_kernel(n, initializer);

	convolutionKernel << <gridDim, blockDim >> > (d_images_float, d_output, width, height, outputWidth, outputHeight, kernel);
	cudaDeviceSynchronize();
	printf("Convolution done\n");

	float* h_output = (float*)malloc(outputWidth * outputHeight * numImages * sizeof(float));
	cudaMemcpy(h_output, d_output, outputWidth * outputHeight * numImages * sizeof(float), cudaMemcpyDeviceToHost);

	// print image after convolution
	int counter = 0;
	printf("First images after convolution\n");
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < outputWidth * outputHeight; j++) {
			std::cout << h_output[j + i * outputWidth * outputHeight] << " ";
			counter++;
		}
		std::cout << std::endl;
	}

	printf("Total number of pixels: %d\n", counter);
	

	// Free memory
	free(h_output);
	cudaFree(d_output);
}
