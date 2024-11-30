#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "load_images.cu"
#include "preprocess_images.cu"
#include "verify_images.cu"
#include "convolution.cu"
//#include "max_pooling.cu"
#include "activations.cuh"
#include "dense_layer.cu"
#include "backpropagation.cu"
#include "sgd.cu"
#include "loss.cu"

#define IMG_SIZE 32*32*3 // 32x32x3
#define NUM_IMAGES 10000 // 10000 images per batch
#define DATA_BATCHES 5   // Total number of data batches
#define EPOCHS 10
#define LEARNING_RATE 0.0001f
#define NUM_TEST_SAMPLES 10 // Number of test samples to evaluate
#define BATCH_SIZE 32          // Add batch processing
#define WEIGHT_DECAY 0.0001f   // Add weight decay to prevent overfitting

// Function to get predicted class (returns index of maximum value)
__global__ void getPredictedClass(float* softmax_output, int* predictions, int batchSize, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        float maxVal = softmax_output[idx * numClasses];
        int maxIdx = 0;
        
        for (int i = 1; i < numClasses; ++i) {
            float val = softmax_output[idx * numClasses + i];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        predictions[idx] = maxIdx;
    }
}

// Function to evaluate model on test samples
void evaluateModel(ConvolutionLayer& conv1, DenseLayer& dense1, DenseLayer& dense2, 
                  float* test_images, float* test_labels, int numSamples) {
    
    // Forward pass
    float* conv_output = conv1.forward(test_images);
    float* dense_output1 = dense1.forward(conv_output);
    float* dense_output2 = dense2.forward(dense_output1);

    // Apply softmax
    float* softmax_output;
    cudaMalloc(&softmax_output, numSamples * 10 * sizeof(float));
    dim3 blockDim_softmax(256);
    dim3 gridDim_softmax((numSamples + blockDim_softmax.x - 1) / blockDim_softmax.x);
    softmaxKernel<<<gridDim_softmax, blockDim_softmax>>>(dense_output2, softmax_output, numSamples, 10);
    
    // Get predictions
    int* d_predictions;
    cudaMalloc(&d_predictions, numSamples * sizeof(int));
    getPredictedClass<<<(numSamples + 255) / 256, 256>>>(softmax_output, d_predictions, numSamples, 10);
    
    // Copy predictions and labels to host
    int* h_predictions = new int[numSamples];
    float* h_labels = new float[numSamples];
    cudaMemcpy(h_predictions, d_predictions, numSamples * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labels, test_labels, numSamples * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "\nTest Results:" << std::endl;
    std::cout << "Sample\tPredicted\tActual" << std::endl;
    for (int i = 0; i < numSamples; ++i) {
        std::cout << i << "\t" << h_predictions[i] << "\t\t" << (int)h_labels[i] << std::endl;
    }

    // Cleanup
    delete[] h_predictions;
    delete[] h_labels;
    cudaFree(softmax_output);
    cudaFree(d_predictions);
}

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

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Forward pass
        float* conv_output = conv1.forward(d_images_float);
        
        // Add ReLU activation after convolution
        int convOutputSize = conv1.getPoolOutputWidth() * conv1.getPoolOutputHeight() * 
                            conv1.getPoolOutputChannels() * NUM_IMAGES;
        dim3 reluBlock(256);
        dim3 reluGrid((convOutputSize + 255) / 256);
        reluActivationKernel<<<reluGrid, reluBlock>>>(conv_output, conv_output, convOutputSize);
        
        float* dense_output1 = dense1.forward(conv_output);
        
        // Add ReLU activation after first dense layer
        int dense1OutputSize = dense1.getOutputSize() * NUM_IMAGES;
        dim3 relu2Block(256);
        dim3 relu2Grid((dense1OutputSize + 255) / 256);
        reluActivationKernel<<<relu2Grid, relu2Block>>>(dense_output1, dense_output1, 
                                                       dense1OutputSize);
        
        float* dense_output2 = dense2.forward(dense_output1);
        
        // Apply softmax activation
        float* softmax_output;
        cudaMalloc(&softmax_output, 10 * NUM_IMAGES * sizeof(float));
        dim3 blockDim_softmax(256);
        dim3 gridDim_softmax((NUM_IMAGES + blockDim_softmax.x - 1) / blockDim_softmax.x);
        softmaxKernel<<<gridDim_softmax, blockDim_softmax>>>(dense_output2, softmax_output, NUM_IMAGES, 10);
        cudaDeviceSynchronize();

        // Calculate and print loss
        float loss = calculateLoss(softmax_output, d_labels_float, 10, NUM_IMAGES);
        std::cout << "Epoch " << epoch + 1 << ": Loss = " << loss << std::endl;

        // Backpropagation for Dense Layer 2
        backpropagate(softmax_output, d_labels_float, d_gradients_dense2, 10, NUM_IMAGES);

        // Update weights and biases for Dense Layer 2 using SGD
        dense2.backward(dense_output1, d_gradients_dense2);
        sgdUpdateWeights(dense2.getWeights(), dense2.getGradWeights(), 64 * 10, LEARNING_RATE);
        sgdUpdateBiases(dense2.getBiases(), dense2.getGradBiases(), 10, LEARNING_RATE);

        // Backpropagation for Dense Layer 1
        float* d_gradients_dense1 = dense1.getGradWeights();
        dense1.backward(conv_output, d_gradients_dense1);
        sgdUpdateWeights(dense1.getWeights(), dense1.getGradWeights(), dense1.getOutputSize() * dense1.getBatchSize(), LEARNING_RATE);
        sgdUpdateBiases(dense1.getBiases(), dense1.getGradBiases(), dense1.getOutputSize(), LEARNING_RATE);

        // Backpropagation for Convolution Layer
        float* d_gradients_conv = conv1.getGradFilters();
        conv1.backward(d_images_float, d_gradients_conv);
        sgdUpdateWeights(conv1.getFilters(), conv1.getGradFilters(), FILTER_SIZE * FILTER_SIZE * conv1.getOutputChannels(), LEARNING_RATE);

        // Free softmax_output
        cudaFree(softmax_output);
    }

    // Test the model on some samples
    std::cout << "\nEvaluating model on test samples..." << std::endl;
    evaluateModel(conv1, dense1, dense2, 
                 d_images_float, // Using first NUM_TEST_SAMPLES images from training set
                 d_labels_float, 
                 NUM_TEST_SAMPLES);

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