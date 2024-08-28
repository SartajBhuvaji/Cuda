#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5


__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int gray = 0;

		for (int c = 0; c < channels; ++c) {
			int index = (y * width + x) * channels + c; // index of pixel in input array
			gray += input[index]; // sum of all channels
		}

		gray /= channels; // average of all channels

		for (int c = 0; c < channels; ++c) {
			int outIndex = (y * width + x) * channels + c; // index of pixel in output array
			output[outIndex] = static_cast<unsigned char>(gray); // set all channels to average
		}
	}
}

__global__ void normalize_image(unsigned char* input, unsigned char* output, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		for (int c = 0; c < channels; ++c) {
			int index = (y * width + x) * channels + c; // index of pixel in input array
			output[index] = input[index] / 255.0; // normalize pixel value
		}
	}
}

void preprocess_data(unsigned char* d_images, unsigned char* d_labels) {

	// Set up grid and block dimensions
	dim3 blockSize(16, 16);
	dim3 gridSize((32 + blockSize.x - 1) / blockSize.x, (32 + blockSize.y - 1) / blockSize.y);

	// Call kernel
	//grayscaleKernel << <gridSize, blockSize >> > (d_images, d_images, 32, 32, 3);
	//cudaDeviceSynchronize();

	//normalize_image << <gridSize, blockSize >> > (d_images, d_images, 32, 32, 3);
	//cudaDeviceSynchronize();
	
	printf("Data preprocessed\n");
}