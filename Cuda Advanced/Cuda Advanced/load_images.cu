#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>


#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5      // Total number of data batches

void loadBatch(const char* filename, unsigned char* h_images, unsigned char* h_labels, int offset) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Couldn't open file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < NUM_IMAGES; i++) {
        fread(&h_labels[offset + i], 1, 1, file);                     // Read label
        fread(&h_images[(offset + i) * IMG_SIZE], 1, IMG_SIZE, file); // Read image    
    }
    printf("Loaded %s\n", filename);
    fclose(file);
}

void allocateMemory(unsigned char* d_images, unsigned char* d_labels) {
    cudaMalloc(&d_images, IMG_SIZE * NUM_IMAGES * DATA_BATCHES);
    cudaMalloc(&d_labels, NUM_IMAGES * DATA_BATCHES);
}


void transferToCUDA(unsigned char* d_images, unsigned char* h_images, unsigned char* d_labels, unsigned char* h_labels) {
    cudaMemcpy(d_images, h_images, IMG_SIZE * NUM_IMAGES * DATA_BATCHES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, NUM_IMAGES * DATA_BATCHES, cudaMemcpyHostToDevice);

}

std::tuple<unsigned char*, unsigned char*> load_data() {
    unsigned char* d_images = nullptr;
    unsigned char* d_labels = nullptr;

    unsigned char* h_images = (unsigned char*)malloc(IMG_SIZE * NUM_IMAGES * DATA_BATCHES);
    unsigned char* h_labels = (unsigned char*)malloc(NUM_IMAGES * DATA_BATCHES);

    if (h_images == nullptr || h_labels == nullptr) {
        printf("Error: Memory allocation failed\n");
        exit(1);
    }

    cudaMalloc(&d_images, IMG_SIZE * NUM_IMAGES * DATA_BATCHES);
    cudaMalloc(&d_labels, NUM_IMAGES * DATA_BATCHES);

    if (d_images == nullptr || d_labels == nullptr) {
        printf("Error: CUDA memory allocation failed\n");
        exit(1);
    }

    const char* base_path = "C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\cifar-10\\data_batch_";
    char full_path[256];

    // Load all data batches
    for (int i = 1; i <= DATA_BATCHES; i++) {
        snprintf(full_path, sizeof(full_path), "%s%d.bin", base_path, i);
        loadBatch(full_path, h_images, h_labels, (i - 1) * NUM_IMAGES);
    }

	transferToCUDA(d_images, h_images, d_labels, h_labels);
    printf("Data loaded and transferred to CUDA\n");

    free(h_images);
    free(h_labels);

	// Check Cuda errors
	cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in load_images: %s\n", cudaGetErrorString(error));
    }

    return std::make_tuple(d_images, d_labels);
}


// Function to return the n th image from the batch and the nth label
//std::tuple<unsigned char*, unsigned char> get_image(unsigned char* d_images, unsigned char* d_labels, int n) {
//	unsigned char* h_images = (unsigned char*)malloc(IMG_SIZE);
//	unsigned char h_label;
//
//	cudaMemcpy(h_images, d_images + n * IMG_SIZE, IMG_SIZE, cudaMemcpyDeviceToHost);
//	cudaMemcpy(&h_label, d_labels + n, 1, cudaMemcpyDeviceToHost);
//
//	return std::make_tuple(h_images, h_label);
//}