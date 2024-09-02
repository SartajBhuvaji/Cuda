#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>


#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\load_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\preprocess_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\verify_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\convolution.cu>


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
    printf("First images\n");
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            std::cout << h_images_float[j + i * IMG_SIZE] << " ";
            counter++;
        }
        std::cout << std::endl;
    }
    printf("Total number of pixels: %d\n", counter);

    // Convert and display the first image
    //convertAndDisplayImage(h_images_float, h_labels_float);

    float* conv_image, * conv_label, * conv_kernel;
    ConvolutionResult conv_result = convolution(d_images_float, 32, 32, NUM_IMAGES * DATA_BATCHES);


    // TODO : COLLECT THE OUTPUT FROM CONVOLUTION
    free(conv_result.output);
    free(conv_result.kernel);
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