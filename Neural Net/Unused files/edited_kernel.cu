#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>

// Include all necessary files
#include "load_images.cu"
#include "preprocess_images.cu"
#include "verify_images.cu"
#include "convolution.cu"
#include "dense_layer.cu"
#include "backpropagation.cu"
#include "sgd.cu"
#include "loss.cu"
#include "activations.cu"

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5   // Total number of data batches
#define EPOCHS 100
#define LEARNING_RATE 0.01f

void gpu_mem_info() {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << "\nGPU memory usage: used = " << used_db / 1024.0 / 1024.0 << "MB, free = " << free_db / 1024.0 / 1024.0 << "MB, total = " << total_db / 1024.0 / 1024.0 << "MB" << std::endl;
}

void testModel(ConvolutionLayer& conv1, float* d_images_float, float* d_labels_float, DenseLayer& dense1, DenseLayer& dense2, int numSamples) {
    // Forward pass through the network
    float* conv_output = conv1.forward(d_images_float);
    float* dense_output1 = dense1.forward(conv_output);
    float* dense_output2 = dense2.forward(dense_output1);

    // Apply softmax activation
    float* softmax_output;
    cudaMalloc(&softmax_output, 10 * numSamples * sizeof(float));
    dim3 blockDim_softmax(256);
    dim3 gridDim_softmax((numSamples + blockDim_softmax.x - 1) / blockDim_softmax.x);
    softmaxKernel<<<gridDim_softmax, blockDim_softmax>>>(dense_output2, softmax_output, numSamples, 10);
    cudaDeviceSynchronize();

    // Copy the result back to host for visualization
    float* h_softmax_output = (float*)malloc(10 * numSamples * sizeof(float));
    cudaMemcpy(h_softmax_output, softmax_output, 10 * numSamples * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy the actual labels back to host
    float* h_labels = new float[numSamples * 10];
    cudaMemcpy(h_labels, d_labels_float, numSamples * 10 * sizeof(float), cudaMemcpyDeviceToHost);

    // Display actual and predicted labels for the first few images
    for (int i = 0; i < numSamples; ++i) {
        int predicted_label = 0;
        float max_prob = h_softmax_output[i * 10];
        for (int j = 1; j < 10; ++j) {
            if (h_softmax_output[i * 10 + j] > max_prob) {
                max_prob = h_softmax_output[i * 10 + j];
                predicted_label = j;
            }
        }

        int actual_label = 0;
        for (int j = 0; j < 10; ++j) {
            if (h_labels[i * 10 + j] == 1.0f) {
                actual_label = j;
                break;
            }
        }

        std::cout << "Sample " << i << ": Actual Label = " << actual_label << ", Predicted Label = " << predicted_label << std::endl;
    }

    free(h_softmax_output);
    delete[] h_labels;
    cudaFree(softmax_output);
}

int main() {
    // Seed for reproducibility
    srand(time(0));

    // Load data
    unsigned char* d_images = nullptr;
    unsigned char* d_labels = nullptr;
    std::tie(d_images, d_labels) = load_data();

    // Preprocess and convert to float (one-hot encoded labels)
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

    // Allocate memory for gradients for Dense Layer 2
    float* d_gradients_dense2;
    cudaMalloc(&d_gradients_dense2, 10 * NUM_IMAGES * sizeof(float));

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Apply softmax activation
        float* softmax_output;
        cudaMalloc(&softmax_output, 10 * NUM_IMAGES * sizeof(float));
        dim3 blockDim_softmax(256);
        dim3 gridDim_softmax((NUM_IMAGES + blockDim_softmax.x - 1) / blockDim_softmax.x);
        softmaxKernel<<<gridDim_softmax, blockDim_softmax>>>(dense_output2, softmax_output, NUM_IMAGES, 10);
        cudaDeviceSynchronize();

        // Calculate loss
        float loss = calculateLoss(softmax_output, d_labels_float, 10, NUM_IMAGES);
        if ((epoch + 1) % 10 == 0 || epoch == 0) { // Print loss every 10 epochs and first epoch
            std::cout << "Epoch " << epoch + 1 << ": Loss = " << loss << std::endl;
        }

        // Backpropagation for Dense Layer 2
        backpropagate(softmax_output, d_labels_float, d_gradients_dense2, 10, NUM_IMAGES);

        // Update weights and biases for Dense Layer 2 using SGD
        dense2.backward(dense_output1, d_gradients_dense2);
        sgdUpdateWeights(dense2.getWeights(), dense2.getGradWeights(), 64 * 10, LEARNING_RATE);
        sgdUpdateBiases(dense2.getBiases(), dense2.getGradBiases(), 10, LEARNING_RATE);

        // Backpropagation for Dense Layer 1
        float* d_gradients_dense1 = dense1.getGradWeights(); // Allocate and compute gradients for dense1
        dense1.backward(conv_output, d_gradients_dense1);
        sgdUpdateWeights(dense1.getWeights(), dense1.getGradWeights(), dense1.getOutputSize() * dense1.getBatchSize(), LEARNING_RATE);
        sgdUpdateBiases(dense1.getBiases(), dense1.getGradBiases(), dense1.getOutputSize(), LEARNING_RATE);

        // Backpropagation for Convolution Layer
        float* d_gradients_conv = conv1.getGradFilters(); // Allocate and compute gradients for conv1
        conv1.backward(d_images_float, d_gradients_conv);
        sgdUpdateWeights(conv1.getFilters(), conv1.getGradFilters(), FILTER_SIZE * FILTER_SIZE * conv1.getOutputChannels(), LEARNING_RATE);

        // Free softmax_output
        cudaFree(softmax_output);
    }

    // Test the model
    testModel(conv1, d_images_float, d_labels_float, dense1, dense2, 5);

    // Free resources
    cudaFree(d_gradients_dense2);
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_images_float);
    cudaFree(d_labels_float);
    cudaFree(dense_output1);
    cudaFree(dense_output2);

    return 0;
} 