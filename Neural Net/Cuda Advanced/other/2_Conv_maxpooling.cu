#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define FILTER_SIZE 5 
#define SIGMA 10.0f // Standard deviation of Gaussian filter // higher value will make the image more blurry
#define POOL_SIZE 2 // Pooling size (e.g., 2x2)


// Grayscale conversion
__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int gray = 0;

        for (int c = 0; c < channels; ++c) {
            int index = (y * width + x) * channels + c;
            gray += input[index];
        }

		gray /= channels; // Average value

        for (int c = 0; c < channels; ++c) {
            int outIndex = (y * width + x) * channels + c;
			output[outIndex] = static_cast<unsigned char>(gray); // Set the same value for all channels
        }
    }
}

// Create Gaussian filter
// Shared memory is used to store the filter
// The filter is normalized so that the sum of all elements is 1
__device__ void createFilter(float* sharedFilter, int n, float sigma) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (tx < n && ty < n) {
        float x = tx - (n - 1) / 2.0f;
        float y = ty - (n - 1) / 2.0f;
		float value = expf(-(x * x + y * y) / (2.0f * sigma * sigma)); // Gaussian function
		sharedFilter[ty * n + tx] = value; // Store the filter in shared memory
    }
    __syncthreads();

	// Normalize the filter so that the sum of all elements is 1
    if (tx == 0 && ty == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n * n; ++i) {
            sum += sharedFilter[i];
        }
        for (int i = 0; i < n * n; ++i) {
            sharedFilter[i] /= sum;
        }
    }
	__syncthreads(); // Wait for all threads to finish
}

// Convolution operation
// Apply Gaussian filter to the input image
__global__ void convolutionKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
	
    extern __shared__ float sharedFilter[]; // Shared memory to store the filter 
	// The filter is stored in shared memory so that it can be accessed by all threads in the block

	createFilter(sharedFilter, FILTER_SIZE, SIGMA); // Create Gaussian filter that we will use for convolution

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
		// Apply filter
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int fy = 0; fy < FILTER_SIZE; ++fy) {
                for (int fx = 0; fx < FILTER_SIZE; ++fx) {

                    int imgX = x + fx - FILTER_SIZE / 2; 
                    int imgY = y + fy - FILTER_SIZE / 2;

                    if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                        int index = (imgY * width + imgX) * channels + c;
						sum += input[index] * sharedFilter[fy * FILTER_SIZE + fx]; // Convolution operation 
                    }
                }
            }
            int outIndex = (y * width + x) * channels + c;
            output[outIndex] = static_cast<unsigned char>(fminf(fmaxf(sum, 0.0f), 255.0f));
        }
    }
}

// Max pooling operation
__global__ void maxPoolingKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outWidth = width / POOL_SIZE;
    int outHeight = height / POOL_SIZE;

    if (outX < outWidth && outY < outHeight) {
		// Find maximum value in pooling window

        for (int c = 0; c < channels; ++c) {
            unsigned char maxVal = 0;
            for (int y = 0; y < POOL_SIZE; ++y) {
                for (int x = 0; x < POOL_SIZE; ++x) {
                    int inX = outX * POOL_SIZE + x;
                    int inY = outY * POOL_SIZE + y;
                    if (inX < width && inY < height) {
                        int index = (inY * width + inX) * channels + c;
                        maxVal = max(maxVal, input[index]);
                    }
                }
            }
            int outIndex = (outY * outWidth + outX) * channels + c;
            output[outIndex] = maxVal;
        }
    }
}

int main() {
	// Using OpenCV to read image
    cv::Mat image = cv::imread("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\image.jpg", cv::IMREAD_UNCHANGED);
    if (image.empty()) {
		std::cerr << "Error: Could not read input image." << std::endl; // Check if I put the right path
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
	int channels = image.channels(); // 3 for RGB image

    unsigned char* d_input, * d_output, * d_temp1, * d_temp2;
    size_t imageSize = width * height * channels * sizeof(unsigned char);
	size_t pooledImageSize = (width / POOL_SIZE) * (height / POOL_SIZE) * channels * sizeof(unsigned char); // Size of the output image

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_temp1, imageSize);
    cudaMalloc(&d_temp2, imageSize);
    cudaMalloc(&d_output, pooledImageSize);

    cudaMemcpy(d_input, image.data, imageSize, cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	// Pooled grid size
    dim3 pooledGridSize(((width / POOL_SIZE) + blockSize.x - 1) / blockSize.x, ((height / POOL_SIZE) + blockSize.y - 1) / blockSize.y);

    // Step 1: Grayscale conversion
    grayscaleKernel << <gridSize, blockSize >> > (d_input, d_temp1, width, height, channels);
    cudaDeviceSynchronize();

    // Step 2: Convolution (e.g., Gaussian blur)
	size_t sharedMemSize = FILTER_SIZE * FILTER_SIZE * sizeof(float); // Shared memory size
    convolutionKernel << <gridSize, blockSize, sharedMemSize >> > (d_temp1, d_temp2, width, height, channels);
    cudaDeviceSynchronize();

    // Step 3: Max Pooling
    maxPoolingKernel << <pooledGridSize, blockSize >> > (d_temp2, d_output, width, height, channels);
    cudaDeviceSynchronize();

    // Copy result back to host
    cv::Mat result((height / POOL_SIZE), (width / POOL_SIZE), image.type());
    cudaMemcpy(result.data, d_output, pooledImageSize, cudaMemcpyDeviceToHost);

    // Save output image
    cv::imwrite("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\output.jpg", result);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    cudaFree(d_output);

    std::cout << "Image processing. Output saved as output.jpg" << std::endl;
    return 0;
}
