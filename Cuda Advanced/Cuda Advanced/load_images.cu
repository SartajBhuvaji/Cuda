﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>


#define IMG_SIZE (32*32*3)  // 32x32x3
#define NUM_IMAGES 10000    // 10000 images per batch
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

void verify_GPUload(unsigned char* d_images, unsigned char* d_labels) {
    
    unsigned char* h_images = (unsigned char*)malloc(IMG_SIZE * NUM_IMAGES * DATA_BATCHES);
    unsigned char* h_labels = (unsigned char*)malloc(NUM_IMAGES * DATA_BATCHES);

    // display the first image with its label
    cudaMemcpy(h_images, d_images, IMG_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labels, d_labels, NUM_IMAGES, cudaMemcpyDeviceToHost);

    // view first image
    cv::Mat img(32, 32, CV_8UC3);
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            for (int c = 0; c < 3; c++) {
                img.at<cv::Vec3b>(y, x)[c] = h_images[y * 32 + x + c * 1024];
            }
        }
    }
    cv::imshow("Image", img);

    // display the label
    printf("Label: %d\n", h_labels[0]);
    cv::waitKey(0);

	// Free host memory
	free(h_images);
	free(h_labels);

}

void verify_GPU_batch_load(unsigned char* d_images, unsigned char* d_labels) {
	printf("Verifying GPU batch load\n");
    // load images to CPU
    unsigned char* h_images = (unsigned char*)malloc(IMG_SIZE * NUM_IMAGES * DATA_BATCHES);
    unsigned char* h_labels = (unsigned char*)malloc(NUM_IMAGES * DATA_BATCHES);

	// display the first image with its label for each batch
	for (int i = 0; i < DATA_BATCHES; i++) {
		cudaMemcpy(h_images, d_images + i * IMG_SIZE * NUM_IMAGES, IMG_SIZE, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_labels, d_labels + i * NUM_IMAGES, NUM_IMAGES, cudaMemcpyDeviceToHost);

		// Check if image is RGB or grayscale
		bool isRGB = false;

		for (int i = 0; i < IMG_SIZE; i++) {
			if (h_images[i] != h_images[i + 1024] || h_images[i] != h_images[i + 2048]) {
				isRGB = true;
				break;
			}
		}
        
		int image_dim = isRGB ? 3 : 1;
		printf("Image dimension: %d\n", image_dim); 

		// view first image
		cv::Mat img(32, 32, CV_8UC3);
		for (int y = 0; y < 32; y++) {
			for (int x = 0; x < 32; x++) {
				for (int c = 0; c < image_dim; c++) {
					img.at<cv::Vec3b>(y, x)[c] = h_images[y * 32 + x + c * 1024];
				}
			}
		}
		cv::imshow("Image", img);
		printf("Label: %d\n", h_labels[0]);
        //cv::imwrite("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\image" + std::to_string(i) + ".jpg", img);
		cv::waitKey(5000);
	}
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

    cudaMemcpy(d_images, h_images, IMG_SIZE * NUM_IMAGES * DATA_BATCHES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, NUM_IMAGES * DATA_BATCHES, cudaMemcpyHostToDevice);
    printf("Data loaded and transferred to CUDA\n");

    free(h_images);
    free(h_labels);

    return std::make_tuple(d_images, d_labels);
}