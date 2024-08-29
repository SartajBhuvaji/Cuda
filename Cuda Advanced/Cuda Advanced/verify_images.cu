#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5      // Total number of data batches

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

		cv::Mat img(32, 32, CV_8UC3);
		for (int y = 0; y < 32; y++) {
			for (int x = 0; x < 32; x++) {
				for (int c = 0; c < image_dim; c++) {
					img.at<cv::Vec3b>(y, x)[c] = h_images[y * 32 + x + c * 1024];
				}
			}
		}

		// print RGB value of the first pixel
		printf("RGB: %d %d %d\n", img.at<cv::Vec3b>(0, 0)[0], img.at<cv::Vec3b>(0, 0)[1], img.at<cv::Vec3b>(0, 0)[2]);
		cv::resize(img, img, cv::Size(250, 250));
		cv::imshow("Image", img);
		printf("Label: %d\n", h_labels[0]);
		//cv::imwrite("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\image" + std::to_string(i) + ".jpg", img);
		cv::waitKey(5000);
	}
}