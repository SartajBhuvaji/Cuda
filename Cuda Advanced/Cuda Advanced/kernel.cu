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


void gpu_mem_info() {

    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << "\nGPU memory usage: used = " << used_db / 1024.0 / 1024.0 << "MB, free = " << free_db / 1024.0 / 1024.0 << "MB, total = " << total_db / 1024.0 / 1024.0 << "MB" << std::endl;
}


void convertAndDisplayImage(float* h_images_float, int imageIndex, int width, int height) {
    // Create a cv::Mat object for the grayscale image
    cv::Mat grayscaleImage(height, width, CV_32F);

    // Copy the image data into the cv::Mat object
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            grayscaleImage.at<float>(i, j) = h_images_float[imageIndex * width * height + i * width + j];
        }
    }

    // Normalize the image to 0-255 range
    cv::Mat normalizedImage;
    cv::normalize(grayscaleImage, normalizedImage, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Resize the image to display it larger
    cv::Mat resizedImage;
    cv::resize(normalizedImage, resizedImage, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);

    // Display the image
    cv::imshow("Grayscale Image", resizedImage);
    cv::waitKey(0);  // Wait for a key press
    cv::destroyAllWindows();  // Close the window
}

int main() {
    // Step 1. Load data
    unsigned char* d_images = nullptr;
    unsigned char* d_labels = nullptr;
    std::tie(d_images, d_labels) = load_data();
    if (d_images == nullptr || d_labels == nullptr) {
        std::cerr << "Failed to load data" << std::endl;
        return 1;
    }

    // Convert data to float and normalize
    float* d_images_float = nullptr;
    float* d_labels_float = nullptr;
    preprocessImage(d_images, &d_images_float, d_labels, &d_labels_float);
    gpu_mem_info();

    cudaFree(d_images);
    cudaFree(d_labels);

	// copy from device to host
    float* h_labels_float = (float*)malloc(NUM_IMAGES * DATA_BATCHES * sizeof(float));
	float* h_images_float = (float*)malloc(IMG_SIZE / 3 * NUM_IMAGES * DATA_BATCHES * sizeof(float));

    cudaMemcpy(h_labels_float, d_labels_float, NUM_IMAGES * DATA_BATCHES * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_images_float, d_images_float, IMG_SIZE / 3 * NUM_IMAGES * DATA_BATCHES * sizeof(float), cudaMemcpyDeviceToHost);

    // print the first 10 labels
    for (int i = 0; i < 10; i++) {
        std::cout << h_labels_float[i] << std::endl;
    }

	// print the first image
    int counter = 0;
	printf("First images\n");
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < IMG_SIZE / 3; j++) {
			std::cout << h_images_float[j + i * IMG_SIZE / 3] << " ";
			counter++;
		}
		std::cout << std::endl;
	}
	printf("Total number of pixels: %d\n", counter);

    // Convert and display the first image
    convertAndDisplayImage(h_images_float, 0, 32, 32);




    // Free the allocated memory
    cudaFree(d_images_float);
    cudaFree(d_labels_float);
    free(h_labels_float);

    return 0;
}

    
	    /*

    float* d_images_gray_norm;
    float* d_labels_float;
    cudaMalloc(&d_images_gray_norm, IMG_SIZE / 3 * NUM_IMAGES * DATA_BATCHES * sizeof(float));
    cudaMalloc(&d_labels_float, NUM_IMAGES * DATA_BATCHES * sizeof(float));

    preprocessImages(d_images, d_images_gray_norm, d_labels, d_labels_float);
    verifyGrayscaleConversion(d_images_gray_norm, d_labels_float);

	// Free memory on gpu
	/*cudaFree(d_images); 
	cudaFree(d_labels); 


    float* d_output;
    cudaMalloc(&d_output, (IMG_WIDTH - 2) * (IMG_HEIGHT - 2) * NUM_IMAGES * DATA_BATCHES * sizeof(float));
    perform_convolution(d_images_gray_norm, d_labels_float, NUM_IMAGES * DATA_BATCHES);

    // Verify grayscale conversion, normalization, and convolution
    verify_grayscale_normalization(d_images_gray_norm, d_labels_float, NUM_IMAGES * DATA_BATCHES);

    // Clean up
    cudaFree(d_output);

    */