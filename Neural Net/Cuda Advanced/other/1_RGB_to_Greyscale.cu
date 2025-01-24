//  Cuda kernel to change image to grayscale

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2\opencv.hpp> //SETUP : https://www.youtube.com/watch?v=x5EWlNQ6z5w&t=0s
#include <iostream>


// kernel to change image to grayscale
// This kernel effectively turns a color image into a grayscale image by 
// setting each pixel's color channels to the average of its original color channel values.
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


int main() {
    // Read input image
    cv::Mat image = cv::imread("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\image.jpeg", cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Error: Could not read input image." << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
	int channels = image.channels(); // 3 for BGR image

    // Allocate memory on device
    unsigned char* d_input, * d_output;
    size_t imageSize = width * height * channels * sizeof(unsigned char);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    // Copy input image to device
	cudaMemcpy(d_input, image.data, imageSize, cudaMemcpyHostToDevice); // use image.data to get pointer to image data

    // Set up grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    grayscaleKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, channels);

    // Copy result back to host
    cv::Mat result(height, width, image.type());
    cudaMemcpy(result.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Save output image
    cv::imwrite("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\output.jpg", result);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Greyscale applied to output.jpg" << std::endl;
    return 0;

}

