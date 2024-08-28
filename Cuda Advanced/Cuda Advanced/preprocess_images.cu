﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5



__global__ void convert_to_float(unsigned char* d_images, float* d_images_float, unsigned char* d_label, float* d_label_float) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	d_images_float[idx] = (float)d_images[idx] * 0.0f; // Normalize to 0-1

	// convert labels to float
	d_label_float[idx] = (float)d_label[idx];
}

__global__ void convert_to_unsigened_char(float* d_images_float, unsigned char* d_images_new, 
								float* d_labels_float, unsigned char* d_labels_new) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	d_images_new[idx] = (unsigned char)(d_images_float[idx] * 255.0f); // Normalize to 0-1

	// convert labels to float
	d_labels_new[idx] = (unsigned char)d_labels_float[idx];
}



void preprocess_images(unsigned char*& d_images, unsigned char*& d_labels) {
    // Step1. Convert to float
    float* d_images_float, * d_labels_float;
    cudaMalloc(&d_images_float, IMG_SIZE * NUM_IMAGES * DATA_BATCHES * sizeof(float));
    cudaMalloc(&d_labels_float, NUM_IMAGES * DATA_BATCHES * sizeof(float));

    int totalThreads = IMG_SIZE * NUM_IMAGES * DATA_BATCHES;
    int blockSize = 256;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    convert_to_float << <gridSize, blockSize >> > (d_images, d_images_float, d_labels, d_labels_float);
    cudaDeviceSynchronize();
    printf("Converted to float\n");

    // Step2. Convert back to unsigned char
    unsigned char* d_images_new, * d_labels_new;
    cudaMalloc(&d_images_new, IMG_SIZE * NUM_IMAGES * DATA_BATCHES);
    cudaMalloc(&d_labels_new, NUM_IMAGES * DATA_BATCHES);

    convert_to_unsigened_char << <gridSize, blockSize >> > (d_images_float, d_images_new, d_labels_float, d_labels_new);
    cudaDeviceSynchronize();

    // Free old memory and update pointers
    cudaFree(d_images);
    cudaFree(d_labels);
    d_images = d_images_new;
    d_labels = d_labels_new;

    // Free intermediate float arrays
    cudaFree(d_images_float);
    cudaFree(d_labels_float);

    printf("Preprocessing done\n");
}