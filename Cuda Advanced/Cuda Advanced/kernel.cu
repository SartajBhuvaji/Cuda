#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>

#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\load_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\preprocess_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\verify_images.cu>




#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5      // Total number of data batches


int main() {
    // Step 1. Load data
    float* d_images_float = nullptr;
    unsigned char* d_images = nullptr;
    unsigned char* d_labels = nullptr;
    std::tie(d_images, d_labels) = load_data();

    if (d_images == nullptr || d_labels == nullptr) {
        std::cerr << "Failed to load data" << std::endl;
        return 1;
    }

    // Verify data
    verify_GPU_batch_load(d_images, d_labels);


    float* d_images_gray_norm;
    float* d_labels_float;
    cudaMalloc(&d_images_gray_norm, IMG_SIZE / 3 * NUM_IMAGES * DATA_BATCHES * sizeof(float));
    cudaMalloc(&d_labels_float, NUM_IMAGES * DATA_BATCHES * sizeof(float));

    preprocessImages(d_images, d_images_gray_norm, d_labels, d_labels_float);
    verifyGrayscaleConversion(d_images_gray_norm, d_labels_float);

	// Free memory on gpu
	cudaFree(d_images); 
	cudaFree(d_labels); 


    return 0;
}