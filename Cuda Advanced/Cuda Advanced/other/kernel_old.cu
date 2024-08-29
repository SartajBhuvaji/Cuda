#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>


#define IMG_SIZE 3072 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch

unsigned char* d_images, * d_labels; // device pointers

void allocateMemory() {
    cudaMalloc(&d_images, IMG_SIZE * NUM_IMAGES * 5); // 5 data batches
    cudaMalloc(&d_labels, NUM_IMAGES * 5);
}

void loadBatch(const char* filename, unsigned char* h_images, unsigned char* h_labels, int offset) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Couldn't open file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < NUM_IMAGES; i++) {
        fread(&h_labels[offset + i], 1, 1, file);               // Read label
        fread(&h_images[(offset + i) * IMG_SIZE], 1, IMG_SIZE, file); // Read image

        
		// Check image and label
		//if (i == 1) {
		//	cv::Mat img(32, 32, CV_8UC3);
		//	for (int y = 0; y < 32; y++) {
		//		for (int x = 0; x < 32; x++) {
		//			for (int c = 0; c < 3; c++) {
		//				img.at<cv::Vec3b>(y, x)[c] = h_images[(offset + i) * IMG_SIZE + y * 32 + x + c * 1024];
		//			}
		//		}
		//	}
		//	// show image in 100x 100 window for 15 sec
		//	cv::namedWindow("Image", cv::WINDOW_NORMAL);
		//	cv::resizeWindow("Image", 100, 100);
		//	cv::imshow("Image", img);
		//	std::cout << "Label of image is : " << (int)h_labels[offset + i] << std::endl;
        //  cv::waitKey(15000);
		//}
     
    }

    fclose(file);
}

void transferToCUDA(unsigned char* h_images, unsigned char* h_labels) {
    cudaMemcpy(d_images, h_images, IMG_SIZE * NUM_IMAGES * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, NUM_IMAGES * 5, cudaMemcpyHostToDevice);
}


int main() {
    unsigned char* h_images = (unsigned char*)malloc(IMG_SIZE * NUM_IMAGES * 5); // host images
    unsigned char* h_labels = (unsigned char*)malloc(NUM_IMAGES * 5); // host labels

    allocateMemory();

    loadBatch("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\cifar-10\\data_batch_1.bin", h_images, h_labels, 0);
    loadBatch("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\cifar-10\\data_batch_2.bin", h_images, h_labels, NUM_IMAGES);
    loadBatch("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\cifar-10\\data_batch_3.bin", h_images, h_labels, 2 * NUM_IMAGES);
    loadBatch("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\cifar-10\\data_batch_4.bin", h_images, h_labels, 3 * NUM_IMAGES);
    loadBatch("C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\cifar-10\\data_batch_5.bin", h_images, h_labels, 4 * NUM_IMAGES);

    transferToCUDA(h_images, h_labels);

    // Free host memory
    free(h_images);
    free(h_labels);

	// Sleep for 2 minutes to give you time to check memory usage
	//std::this_thread::sleep_for(std::chrono::seconds(120));
    
    // Use the data on the GPU for processing...


    // Free device memory when done
    cudaFree(d_images);
    cudaFree(d_labels);

    return 0;
}

