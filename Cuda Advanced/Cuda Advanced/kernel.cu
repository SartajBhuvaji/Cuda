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
#define LEARNING_RATE 0.0001f  // Increased learning rate
#define NUM_TEST_SAMPLES 10 // Number of test samples to evaluate
#define BATCH_SIZE 128          // Increased batch size
#define WEIGHT_DECAY 0.0001f   // Add weight decay to prevent overfitting
#define MAX_GRADIENT 5.0f       // Increased gradient clipping threshold

// Add this helper function at the top of the file
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

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
void evaluateModel(ConvolutionLayer& conv1, 
                  DenseLayer& dense1, 
                  DenseLayer& dense2,
                  DenseLayer& dense3,
                  DenseLayer& dense4,
                  float* test_images, 
                  float* test_labels, 
                  int numSamples) {
    
    cudaError_t error;
    
    // Forward pass through all layers
    float* conv_output = conv1.forward(test_images);
    error = cudaGetLastError();
    checkCudaError(error, "Conv forward in eval");

    float* dense_output1 = dense1.forward(conv_output);
    error = cudaGetLastError();
    checkCudaError(error, "Dense1 forward in eval");

    float* dense_output2 = dense2.forward(dense_output1);
    error = cudaGetLastError();
    checkCudaError(error, "Dense2 forward in eval");

    float* dense_output3 = dense3.forward(dense_output2);
    error = cudaGetLastError();
    checkCudaError(error, "Dense3 forward in eval");

    float* dense_output4 = dense4.forward(dense_output3);
    error = cudaGetLastError();
    checkCudaError(error, "Dense4 forward in eval");

    // Apply softmax
    float* softmax_output;
    cudaMalloc(&softmax_output, numSamples * 10 * sizeof(float));
    dim3 blockDim_softmax(256);
    dim3 gridDim_softmax((numSamples + blockDim_softmax.x - 1) / blockDim_softmax.x);
    softmaxKernel<<<gridDim_softmax, blockDim_softmax>>>(dense_output4, softmax_output, numSamples, 10);
    error = cudaGetLastError();
    checkCudaError(error, "Softmax in eval");
    
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

__global__ void clipGradientsKernel(float* gradients, int size, float max_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (gradients[idx] > max_value) gradients[idx] = max_value;
        if (gradients[idx] < -max_value) gradients[idx] = -max_value;
    }
}

void clipGradients(float* d_gradients, int size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    clipGradientsKernel<<<grid, block>>>(d_gradients, size, MAX_GRADIENT);
    cudaDeviceSynchronize();
}

__global__ void normalizeInputsKernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = input[idx] / 255.0f;  // Normalize to [0,1]
    }
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

    // Normalize inputs
    dim3 normBlock(256);
    dim3 normGrid((NUM_IMAGES * IMG_SIZE + normBlock.x - 1) / normBlock.x);
    normalizeInputsKernel<<<normGrid, normBlock>>>(d_images_float, NUM_IMAGES * IMG_SIZE);
    cudaDeviceSynchronize();

    // Create and apply convolution layer
    ConvolutionLayer conv1(32, 32, 3, NUM_IMAGES);
    float* conv_output = conv1.forward(d_images_float);

    // Calculate the actual output size from convolution layer
    int conv_output_size = conv1.getPoolOutputWidth() * conv1.getPoolOutputHeight() * 
                          conv1.getPoolOutputChannels();

    // Create and apply dense layers with correct input sizes
    DenseLayer dense1(conv_output_size, 128, NUM_IMAGES);
    float* dense_output1 = dense1.forward(conv_output);

    DenseLayer dense2(128, 64, NUM_IMAGES);
    float* dense_output2 = dense2.forward(dense_output1);

    DenseLayer dense3(64, 32, NUM_IMAGES);
    float* dense_output3 = dense3.forward(dense_output2);

    DenseLayer dense4(32, 10, NUM_IMAGES); // Final layer with 10 classes
    float* dense_output4 = dense4.forward(dense_output3);

    // Allocate memory for gradients for final Dense Layer
    float* d_gradients_dense4;
    cudaMalloc(&d_gradients_dense4, 10 * NUM_IMAGES * sizeof(float));

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
        
        // Dense layer 1 forward + ReLU
        float* dense_output1 = dense1.forward(conv_output);
        int dense1OutputSize = dense1.getOutputSize() * NUM_IMAGES;
        dim3 relu1Grid((dense1OutputSize + 255) / 256);
        reluActivationKernel<<<relu1Grid, reluBlock>>>(dense_output1, dense_output1, dense1OutputSize);
        
        // Dense layer 2 forward + ReLU
        float* dense_output2 = dense2.forward(dense_output1);
        int dense2OutputSize = dense2.getOutputSize() * NUM_IMAGES;
        dim3 relu2Grid((dense2OutputSize + 255) / 256);
        reluActivationKernel<<<relu2Grid, reluBlock>>>(dense_output2, dense_output2, dense2OutputSize);
        
        // Dense layer 3 forward + ReLU
        float* dense_output3 = dense3.forward(dense_output2);
        int dense3OutputSize = dense3.getOutputSize() * NUM_IMAGES;
        dim3 relu3Grid((dense3OutputSize + 255) / 256);
        reluActivationKernel<<<relu3Grid, reluBlock>>>(dense_output3, dense_output3, dense3OutputSize);
        
        // Dense layer 4 forward
        float* dense_output4 = dense4.forward(dense_output3);
        
        // Apply softmax activation
        float* softmax_output;
        cudaMalloc(&softmax_output, 10 * NUM_IMAGES * sizeof(float));
        dim3 blockDim_softmax(256);
        dim3 gridDim_softmax((NUM_IMAGES + blockDim_softmax.x - 1) / blockDim_softmax.x);
        softmaxKernel<<<gridDim_softmax, blockDim_softmax>>>(dense_output4, softmax_output, NUM_IMAGES, 10);
        cudaDeviceSynchronize();

        // Calculate and print loss
        float loss = calculateLoss(softmax_output, d_labels_float, 10, NUM_IMAGES);
        std::cout << "Epoch " << epoch + 1 << ": Loss = " << loss << std::endl;

        // Backpropagation
        backpropagate(softmax_output, d_labels_float, d_gradients_dense4, 10, NUM_IMAGES);
        
        // Dense Layer 4 backward
        dense4.backward(dense_output3, d_gradients_dense4);
        clipGradients(dense4.getGradWeights(), 32 * 10);
        sgdUpdateWeights(dense4.getWeights(), dense4.getGradWeights(), dense4.getVelocityWeights(),
                        32 * 10, LEARNING_RATE, 0.9f);
        sgdUpdateBiases(dense4.getBiases(), dense4.getGradBiases(), 10, LEARNING_RATE);

        // Dense Layer 3 backward
        dense3.backward(dense_output2, dense4.getInputGradients());
        clipGradients(dense3.getGradWeights(), 64 * 32);
        sgdUpdateWeights(dense3.getWeights(), dense3.getGradWeights(), dense3.getVelocityWeights(),
                        64 * 32, LEARNING_RATE, 0.9f);
        sgdUpdateBiases(dense3.getBiases(), dense3.getGradBiases(), 32, LEARNING_RATE);

        // Dense Layer 2 backward
        dense2.backward(dense_output1, dense3.getInputGradients());
        clipGradients(dense2.getGradWeights(), 128 * 64);
        sgdUpdateWeights(dense2.getWeights(), dense2.getGradWeights(), dense2.getVelocityWeights(),
                        128 * 64, LEARNING_RATE, 0.9f);
        sgdUpdateBiases(dense2.getBiases(), dense2.getGradBiases(), 64, LEARNING_RATE);

        // Dense Layer 1 backward
        dense1.backward(conv_output, dense2.getInputGradients());
        clipGradients(dense1.getGradWeights(), conv_output_size * 128);
        sgdUpdateWeights(dense1.getWeights(), dense1.getGradWeights(), dense1.getVelocityWeights(),
                        conv_output_size * 128, LEARNING_RATE, 0.9f);
        sgdUpdateBiases(dense1.getBiases(), dense1.getGradBiases(), 128, LEARNING_RATE);

        // Add error checking
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error in training loop: %s\n", cudaGetErrorString(error));
            break;
        }

        // Free intermediate gradients
        cudaFree(d_gradients_dense4);
    }

    // Update evaluation function call
    std::cout << "\nEvaluating model on test samples..." << std::endl;
    evaluateModel(conv1, dense1, dense2, dense3, dense4,
                 d_images_float,
                 d_labels_float,
                 NUM_TEST_SAMPLES);

    // Free resources
    cudaFree(d_gradients_dense4);
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_images_float);
    cudaFree(d_labels_float);
    cudaFree(dense_output1);
    cudaFree(dense_output2);
    cudaFree(dense_output3);
    cudaFree(dense_output4);
    cudaFree(conv_output);

    return 0;
}