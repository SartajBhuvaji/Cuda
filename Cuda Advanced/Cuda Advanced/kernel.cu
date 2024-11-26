#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\load_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\preprocess_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\verify_images.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\convolution.cu>
//#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\max_pooling.cu>
//#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\activations.cu>
#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\dense_layer.cu>


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



int main() {
    // Load data
    unsigned char* d_images = nullptr;
    unsigned char* d_labels = nullptr;
    std::tie(d_images, d_labels) = load_data();

    // Preprocess and convert to float
    float* d_images_float = nullptr;
    float* d_labels_float = nullptr;
    preprocessImage(d_images, &d_images_float, d_labels, &d_labels_float);

    // Create and apply convolution layer
    ConvolutionLayer conv1(32, 32, 3, NUM_IMAGES);
    float* conv_output = conv1.forward(d_images_float);

    // Create and apply dense layers
    DenseLayer dense1(conv1.getPoolOutputWidth() * conv1.getPoolOutputHeight() * conv1.getPoolOutputChannels(), 64, NUM_IMAGES);
    float* dense_output1 = dense1.forward(conv_output);

    DenseLayer dense2(64, 10, NUM_IMAGES); // Assuming 10 classes for classification
    float* dense_output2 = dense2.forward(dense_output1);

    // Apply softmax activation
    float* softmax_output;
    cudaMalloc(&softmax_output, 10 * NUM_IMAGES * sizeof(float));
    dim3 blockDim(256);
    dim3 gridDim((NUM_IMAGES + blockDim.x - 1) / blockDim.x);
    softmaxKernel<<<gridDim, blockDim>>>(dense_output2, softmax_output, NUM_IMAGES, 10);
    cudaDeviceSynchronize();

    // Copy the result back to host for visualization
    float* h_softmax_output = (float*)malloc(10 * NUM_IMAGES * sizeof(float));
    cudaMemcpy(h_softmax_output, softmax_output, 10 * NUM_IMAGES * sizeof(float), cudaMemcpyDeviceToHost);

    // Display actual and predicted labels for the first few images
    unsigned char* h_labels = (unsigned char*)malloc(NUM_IMAGES);
    cudaMemcpy(h_labels, d_labels, NUM_IMAGES, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; ++i) {
        int predicted_label = 0;
        float max_prob = h_softmax_output[i * 10];
        for (int j = 1; j < 10; ++j) {
            if (h_softmax_output[i * 10 + j] > max_prob) {
                max_prob = h_softmax_output[i * 10 + j];
                predicted_label = j;
            }
        }
        std::cout << "Image " << i << ": Actual Label = " << (int)h_labels[i] << ", Predicted Label = " << predicted_label << std::endl;
    }

    // Free resources
    free(h_softmax_output);
    free(h_labels);
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_images_float);
    cudaFree(d_labels_float);
    cudaFree(dense_output1);
    cudaFree(dense_output2);
    cudaFree(softmax_output);

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