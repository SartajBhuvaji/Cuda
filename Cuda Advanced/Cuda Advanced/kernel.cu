#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>


#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\load_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\preprocess_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\verify_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\convolution.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\max_pooling.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\activations.cu>

//#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\dense_layers.cu>



#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5   // Total number of data batches


void gpu_mem_info() {

    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << "\nGPU memory usage: used = " << used_db / 1024.0 / 1024.0 << "MB, free = " << free_db / 1024.0 / 1024.0 << "MB, total = " << total_db / 1024.0 / 1024.0 << "MB" << std::endl;
}


void convertAndDisplayImage(float* h_images_float, float* h_labels_float) {
    cv::Mat img(32, 32, CV_8UC3);
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            for (int c = 0; c < 3; c++) {
                img.at<cv::Vec3b>(y, x)[c] = static_cast<char>(static_cast<int>(h_images_float[y * 32 + x + c * 1024] * 225.0f));
            }
        }
    }
    // print RGB value of the first pixel
    printf("RGB: %d %d %d\n", img.at<cv::Vec3b>(0, 0)[0], img.at<cv::Vec3b>(0, 0)[1], img.at<cv::Vec3b>(0, 0)[2]);
    cv::resize(img, img, cv::Size(250, 250));
    cv::imshow("Image", img);
    printf("Label: %d\n", h_labels_float[0]);
    cv::waitKey(5000);
}


void convertAndDisplayImage_old(float* h_images_float, int imageIndex, int width, int height) {
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

    cv::Mat resizedImage;
    cv::resize(normalizedImage, resizedImage, cv::Size(720, 720), 0, 0, cv::INTER_NEAREST);
    cv::imshow("Grayscale Image", resizedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    // Step 1. Load data
    unsigned char* d_images = nullptr;
    unsigned char* d_labels = nullptr;
    std::tie(d_images, d_labels) = load_data();
    if (d_images == nullptr || d_labels == nullptr) {
        std::cerr << "Failed to load data" << std::endl;
        return -1;
    }

    printf("Priting values just after load_data()\n");
    unsigned char* h_images = (unsigned char*)malloc(IMG_SIZE * NUM_IMAGES * DATA_BATCHES);
    cudaMemcpy(h_images, d_images, IMG_SIZE * NUM_IMAGES * DATA_BATCHES, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100; i++) {
        printf("%d ", (int)h_images[i]);
    }
    printf("\n");

    // Convert data to float and normalize
    float* d_images_float = nullptr;
    float* d_labels_float = nullptr;
    preprocessImage(d_images, &d_images_float, d_labels, &d_labels_float);

    gpu_mem_info();

    cudaFree(d_images);
    cudaFree(d_labels);

    // copy from device to host
    float* h_labels_float = (float*)malloc(NUM_IMAGES * DATA_BATCHES * sizeof(float));
    //float* h_images_float = (float*)malloc(IMG_SIZE / 3 * NUM_IMAGES * DATA_BATCHES * sizeof(float));
    float* h_images_float = (float*)malloc(IMG_SIZE * NUM_IMAGES * DATA_BATCHES * sizeof(float));

    cudaMemcpy(h_labels_float, d_labels_float, NUM_IMAGES * DATA_BATCHES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_images_float, d_images_float, IMG_SIZE * NUM_IMAGES * DATA_BATCHES * sizeof(float), cudaMemcpyDeviceToHost);

    //cudaMemcpy(h_images_float, d_images_float, IMG_SIZE / 3 * NUM_IMAGES * DATA_BATCHES * sizeof(float), cudaMemcpyDeviceToHost);

    // print the first 10 labels
    for (int i = 0; i < 10; i++) {
        std::cout << h_labels_float[i] << std::endl;
    }

    // print the first image
    int counter = 0;
    printf("First image before convolution\n");
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            std::cout << h_images_float[j + i * IMG_SIZE] << " ";
            counter++;
        }
        std::cout << std::endl;
    }
    printf("Total number of pixels: %d\n", counter);

    //  CONVOLUTION
    int inputWidth = 32, inputHeight = 32, inputChannels = 3;

    ConvolutionLayer conv1(inputWidth, inputHeight, inputChannels, NUM_IMAGES);
    // Perform forward pass
    float* conv1d_output_conv = conv1.forward(d_images_float);

    // Allocate host memory for the output
     int conv1outputWidth = conv1.getOutputWidth();
    int conv1outputHeight = conv1.getOutputHeight();
    int conv1outputChannels = conv1.getOutputChannels();
    //float* conv1h_output = (float*)malloc(conv1outputWidth * conv1outputHeight * conv1outputChannels * NUM_IMAGES * sizeof(float));
    /*float* conv1h_conv_filter = (float*)malloc(FILTER_SIZE * FILTER_SIZE * inputChannels * conv1outputChannels * sizeof(float));*/
    //printf("Output width: , Output height: , Output channels: %d %d %d\n", conv1outputWidth, conv1outputHeight, conv1outputChannels);

    // Copy the result back to host
    // cudaMemcpy(conv1h_output, conv1d_output_conv, conv1outputWidth * conv1outputHeight * conv1outputChannels * NUM_IMAGES * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first image after convolution
    counter = 0;
    printf("First image after convolution\n");
    for (int c = 0; c < conv1outputChannels; ++c) {
        for (int i = 0; i < conv1outputHeight; ++i) {
            for (int j = 0; j < conv1outputWidth; ++j) {
                //std::cout << conv1h_output[(c * conv1outputHeight * conv1outputWidth) + (i * conv1outputWidth) + j] << " ";
                counter++;
            }
            // std::cout << std::endl;
        }
        //std::cout << "Channel " << outputChannels << " complete" << std::endl;
    }
    printf("Total number of pixels after conv1: %d\n", counter);


    //MAX POOLING
    MaxPoolingLayer pool1(conv1.getOutputWidth(), conv1.getOutputHeight(), conv1.getOutputChannels(), NUM_IMAGES);
    float* d_pool_output = pool1.forward(conv1d_output_conv);

    int poolOutputWidth = pool1.getOutputWidth();
    int poolOutputHeight = pool1.getOutputHeight();
    int poolOutputChannels = pool1.getOutputChannels();
    float* h_pool_output = (float*)malloc(poolOutputWidth * poolOutputHeight * poolOutputChannels * NUM_IMAGES * sizeof(float));

    printf("\nPOOL 1 resutls");
    printf("\nOutput width: , Output height: , Output channels: %d %d %d\n", poolOutputWidth, poolOutputHeight, poolOutputChannels);

    //cudaMemcpy(h_pool_output, d_pool_output, poolOutputWidth * poolOutputHeight * poolOutputChannels * NUM_IMAGES * sizeof(float), cudaMemcpyDeviceToHost);

	// ACTIVATION
	float* d_activated_output;
	cudaMalloc(&d_activated_output, poolOutputWidth * poolOutputHeight * poolOutputChannels * NUM_IMAGES * sizeof(float));
	applyActivation(d_pool_output, d_activated_output, poolOutputWidth* poolOutputHeight* poolOutputChannels* NUM_IMAGES, "relu");


	//// Copy the result back to host
	//float* h_activated_output = (float*)malloc(poolOutputWidth * poolOutputHeight * poolOutputChannels * NUM_IMAGES * sizeof(float));
	//printf("\nACTIVATION results");
	//cudaMemcpy(h_activated_output, d_activated_output, poolOutputWidth * poolOutputHeight * poolOutputChannels * NUM_IMAGES * sizeof(float), cudaMemcpyDeviceToHost);

	//// Print the first image after convolution
	//counter = 0;
	//printf("\n\nFirst image after activation\n");
 //   for (int c = 0; c < poolOutputChannels; ++c) {
 //       for (int i = 0; i < poolOutputHeight; ++i) {
 //           for (int j = 0; j < poolOutputWidth; ++j) {
 //               std::cout << h_activated_output[(c * poolOutputHeight * poolOutputWidth) + (i * poolOutputWidth) + j] << " ";
 //               counter++;
 //           }
 //           std::cout << std::endl;
 //       }
 //   }






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