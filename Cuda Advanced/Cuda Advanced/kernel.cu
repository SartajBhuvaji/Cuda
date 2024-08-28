﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>

#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\load_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\preprocess_images.cu>

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch

//int main() {
//
//	// Step1. Load data
//	std::tuple<unsigned char*, unsigned char*> data = load_data();
//	unsigned char* d_images = std::get<0>(data);
//	unsigned char* d_labels = std::get<1>(data);
//
//	// Step2. Pre-process data
//	preprocess_data(d_images, d_labels);
//
//	// Step3. Display first image in batch
//
//	unsigned char* h_images = (unsigned char*)malloc(IMG_SIZE * NUM_IMAGES);
//	cudaMemcpy(h_images, d_images, IMG_SIZE, cudaMemcpyDeviceToHost);
//
//
//	// save first 10 images on disc
//	for (int i = 0; i < 10; i++) {
//		cv::Mat img(32, 32, CV_8UC3);
//		for (int y = 0; y < 32; y++) {
//			for (int x = 0; x < 32; x++) {
//				for (int c = 0; c < 3; c++) {
//					img.at<cv::Vec3b>(y, x)[c] = h_images[i * IMG_SIZE + y * 32 + x + c * 1024];
//				}
//			}
//		}
//		cv::imwrite("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\image" + std::to_string(i) + ".jpg", img);
//	}
//
//
//
//}

int main() {
	// Step1. Load data
	std::tuple<unsigned char*, unsigned char*> data = load_data();
	unsigned char* d_images = std::get<0>(data);
	unsigned char* d_labels = std::get<1>(data);

	// Step2. Pre-process data
	preprocess_images(d_images, d_labels);

	// Verify GPU batch load
	verify_GPU_batch_load();	

	return 0;
}

