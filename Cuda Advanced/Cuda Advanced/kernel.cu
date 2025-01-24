#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "cuda_utils.cuh"
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
#define EPOCHS 100        // Number of training epochs
#define LEARNING_RATE 0.0001f  // Increased learning rate
#define NUM_TEST_SAMPLES 80 // Number of test samples to evaluate
#define BATCH_SIZE 128          // Increased batch size
#define WEIGHT_DECAY 0.0001f   // Add weight decay to prevent overfitting
#define MAX_GRADIENT 5.0f       // Increased gradient clipping threshold

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

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < numSamples; ++i) {
        if (h_predictions[i] == (int)h_labels[i]) {
            correct++;
        }
    }
    float accuracy = (float)correct / numSamples;
    std::cout << "\nTest Accuracy = " << accuracy << std::endl;

    // Cleanup
    delete[] h_predictions;
    delete[] h_labels;
    cudaFree(softmax_output);
    cudaFree(d_predictions);
}

int main() {

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

        // Allocate memory for intermediate gradients
        float* d_intermediate_gradients = nullptr;
        cudaError_t error = cudaMalloc(&d_intermediate_gradients, 10 * 10000 * sizeof(float));
        if (error != cudaSuccess) {
            fprintf(stderr, "Failed to allocate intermediate gradients: %s\n", cudaGetErrorString(error));
            // Handle error appropriately
            return -1; // or other error handling
        }

        // Perform backpropagation
        backpropagate(softmax_output, 
                     d_labels_float, 
                     d_gradients_dense4,
                     d_intermediate_gradients,
                     10,    // outputSize
                     10000  // batchSize
        );

        // Dense Layer 4 backward
        dense4.backward(dense_output3, d_gradients_dense4);
        clipGradients(dense4.getGradWeights(), 32 * 10, MAX_GRADIENT);
        sgdUpdateWeights(dense4.getWeights(), dense4.getGradWeights(), dense4.getVelocityWeights(),
                        32 * 10, LEARNING_RATE, 0.9f);
        sgdUpdateBiases(dense4.getBiases(), dense4.getGradBiases(), 10, LEARNING_RATE);

        // Dense Layer 3 backward
        dense3.backward(dense_output2, dense4.getInputGradients());
        clipGradients(dense3.getGradWeights(), 64 * 32, MAX_GRADIENT);
        sgdUpdateWeights(dense3.getWeights(), dense3.getGradWeights(), dense3.getVelocityWeights(),
                        64 * 32, LEARNING_RATE, 0.9f);
        sgdUpdateBiases(dense3.getBiases(), dense3.getGradBiases(), 32, LEARNING_RATE);

        // Dense Layer 2 backward
        dense2.backward(dense_output1, dense3.getInputGradients());
        clipGradients(dense2.getGradWeights(), 128 * 64, MAX_GRADIENT);
        sgdUpdateWeights(dense2.getWeights(), dense2.getGradWeights(), dense2.getVelocityWeights(),
                        128 * 64, LEARNING_RATE, 0.9f);
        sgdUpdateBiases(dense2.getBiases(), dense2.getGradBiases(), 64, LEARNING_RATE);

        // Dense Layer 1 backward
        dense1.backward(conv_output, dense2.getInputGradients());
        clipGradients(dense1.getGradWeights(), conv_output_size * 128, MAX_GRADIENT);
        sgdUpdateWeights(dense1.getWeights(), dense1.getGradWeights(), dense1.getVelocityWeights(),
                        conv_output_size * 128, LEARNING_RATE, 0.9f);
        sgdUpdateBiases(dense1.getBiases(), dense1.getGradBiases(), 128, LEARNING_RATE);

        // Add error checking
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error in training loop: %s\n", cudaGetErrorString(error));
            break;
        }

        // Clean up
        cudaFree(d_intermediate_gradients);
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


// OUTPUT: 
/*
Loaded C:\Users\sbhuv\Desktop\Cuda\Cuda\Cuda Advanced\Cuda Advanced\cifar-10\data_batch_1.bin
Loaded C:\Users\sbhuv\Desktop\Cuda\Cuda\Cuda Advanced\Cuda Advanced\cifar-10\data_batch_2.bin
Loaded C:\Users\sbhuv\Desktop\Cuda\Cuda\Cuda Advanced\Cuda Advanced\cifar-10\data_batch_3.bin
Loaded C:\Users\sbhuv\Desktop\Cuda\Cuda\Cuda Advanced\Cuda Advanced\cifar-10\data_batch_4.bin
Loaded C:\Users\sbhuv\Desktop\Cuda\Cuda\Cuda Advanced\Cuda Advanced\cifar-10\data_batch_5.bin
Preprocessing complete

Epoch 1: Loss = 51.8066
Epoch 2: Loss = 53.2066
Epoch 3: Loss = 52.3066
Epoch 4: Loss = 51.8066
Epoch 5: Loss = 50.4066
Epoch 6: Loss = 49.5066
Epoch 7: Loss = 48.2066
Epoch 8: Loss = 46.9066
Epoch 9: Loss = 46.2066
Epoch 10: Loss = 45.3066
Epoch 11: Loss = 44.3066
Epoch 12: Loss = 44.3066
Epoch 13: Loss = 43.1066
Epoch 14: Loss = 42.5066
Epoch 15: Loss = 41.9066
Epoch 16: Loss = 40.9066
Epoch 17: Loss = 40.2066
Epoch 18: Loss = 39.0066
Epoch 19: Loss = 37.9066
Epoch 20: Loss = 37.3066
Epoch 21: Loss = 36.4066
Epoch 22: Loss = 38.7066
Epoch 23: Loss = 38.0066
Epoch 24: Loss = 37.3066
Epoch 25: Loss = 36.7066
Epoch 26: Loss = 35.6066
Epoch 27: Loss = 34.3066
Epoch 28: Loss = 33.3066
Epoch 29: Loss = 32.1066
Epoch 30: Loss = 31.0066
Epoch 31: Loss = 30.4066
Epoch 32: Loss = 33.1066
Epoch 33: Loss = 32.4066
Epoch 34: Loss = 31.2066
Epoch 35: Loss = 29.8066
Epoch 36: Loss = 28.8066
Epoch 37: Loss = 27.9066
Epoch 38: Loss = 27.1066
Epoch 39: Loss = 26.5066
Epoch 40: Loss = 25.8066
Epoch 41: Loss = 25.0066
Epoch 42: Loss = 28.2066
Epoch 43: Loss = 27.6066
Epoch 44: Loss = 27.0066
Epoch 45: Loss = 26.2066
Epoch 46: Loss = 24.9066
Epoch 47: Loss = 23.7066
Epoch 48: Loss = 22.8066
Epoch 49: Loss = 22.1066
Epoch 50: Loss = 20.9066
Epoch 51: Loss = 19.7066
Epoch 52: Loss = 21.3066
Epoch 53: Loss = 20.7066
Epoch 54: Loss = 19.3066
Epoch 55: Loss = 18.0066
Epoch 56: Loss = 16.9066
Epoch 57: Loss = 15.9066
Epoch 58: Loss = 15.4066
Epoch 59: Loss = 14.7066
Epoch 60: Loss = 13.4066
Epoch 61: Loss = 12.3066
Epoch 62: Loss = 13.8066
Epoch 63: Loss = 12.9066
Epoch 64: Loss = 11.6066
Epoch 65: Loss = 10.5066
Epoch 66: Loss = 9.5066
Epoch 67: Loss = 9.0066
Epoch 68: Loss = 7.6066
Epoch 69: Loss = 7.1066
Epoch 70: Loss = 6.6066
Epoch 71: Loss = 5.5066
Epoch 72: Loss = 7.9066
Epoch 73: Loss = 6.6066
Epoch 74: Loss = 5.2066
Epoch 75: Loss = 4.4066
Epoch 76: Loss = 3.5066
Epoch 77: Loss = 2.6066
Epoch 78: Loss = 2.5643
Epoch 79: Loss = 2.5643
Epoch 80: Loss = 2.5643
Epoch 81: Loss = 2.5643
Epoch 82: Loss = 4.9643
Epoch 83: Loss = 4.0643
Epoch 84: Loss = 2.6643
Epoch 85: Loss = 2.5643
Epoch 86: Loss = 2.5643
Epoch 87: Loss = 2.5643
Epoch 88: Loss = 2.5643
Epoch 89: Loss = 2.5643
Epoch 90: Loss = 2.5643
Epoch 91: Loss = 2.5643
Epoch 92: Loss = 4.9643
Epoch 93: Loss = 3.9643
Epoch 94: Loss = 2.5643
Epoch 95: Loss = 2.5643
Epoch 96: Loss = 2.5643
Epoch 97: Loss = 2.5643
Epoch 98: Loss = 2.5643
Epoch 99: Loss = 2.5643
Epoch 100: Loss = 2.5643

Evaluating model on test samples...

Test Results:
Sample  Predicted       Actual
0       1               1
1       5               5
2       4               4
3       2               2
4       0               0
5       9               9
6       7               7
7       3               3
8       7               2
9       6               6

Test Accuracy = 0.85

C:\Users\sbhuv\Desktop\Cuda\Cuda\Cuda Advanced\x64\Debug\Cuda Advanced.exe (process 22556) exited with code 0 (0x0).
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .
*/