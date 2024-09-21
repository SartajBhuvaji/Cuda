#include <cuda_runtime.h>
#include <cmath>
#include <vector>
//#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\activations.cu>

// Custom CUDA kernel for weight initialization
__global__ void initializeWeightsKernel(float* weights, float* bias, int inputSize, int outputSize, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inputSize * outputSize) {
        // Simple random number generation using XORShift
        unsigned int state = seed + idx;
        state ^= (state << 13);
        state ^= (state >> 17);
        state ^= (state << 5);

        // Generate a random number between -1 and 1
        float random = (float)state / 4294967295.0f * 2.0f - 1.0f;

        // He initialization
        float std_dev = sqrtf(2.0f / inputSize);
        weights[idx] = random * std_dev;
    }

    if (idx < outputSize) {
        // Initialize bias
        unsigned int state = seed + idx + inputSize * outputSize;
        state ^= (state << 13);
        state ^= (state >> 17);
        state ^= (state << 5);

        float random = (float)state / 4294967295.0f * 2.0f - 1.0f;
        float std_dev = sqrtf(2.0f / inputSize);
        bias[idx] = random * std_dev;
    }
}


__global__ void forwardKernel(const float* input, const float* weights, const float* bias,
    float* output, int inputSize, int outputSize, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    if (idx < outputSize && batch < batchSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; ++i) {
            sum += input[batch * inputSize + i] * weights[i * outputSize + idx];
        }
        output[batch * outputSize + idx] = sum + bias[idx];
    }
}

class DenseLayer {
private:
    int inputSize;
    int outputSize;
    int batchSize;
    float* d_weights;
    float* d_bias;
    float* d_output;
    float* d_input;
    const char* activationType;

    void initializeWeightsAndBiases() {
        int blockSize = 256;
        int numBlocks = (inputSize * outputSize + blockSize - 1) / blockSize;
        initializeWeightsKernel << <numBlocks, blockSize >> > (d_weights, d_bias, inputSize, outputSize, 1234);
        cudaDeviceSynchronize();
    }

public:
    DenseLayer(int inSize, int outSize, int bSize, const char* actType)
        : inputSize(inSize), outputSize(outSize), batchSize(bSize), activationType(actType) {
        cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_bias, outputSize * sizeof(float));
        cudaMalloc(&d_output, outputSize * batchSize * sizeof(float));
        cudaMalloc(&d_input, inputSize * batchSize * sizeof(float));
        initializeWeightsAndBiases();
    }

    ~DenseLayer() {
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_output);
        cudaFree(d_input);
    }

    float* forward(const float* input) {
        float h_input[10];
        cudaMemcpy(h_input, input, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Dense layer input (first 10 values):\n");
        for (int i = 0; i < 10; ++i) {
            printf("%f ", h_input[i]);
        }
        printf("\n");

        cudaMemcpy(d_input, input, inputSize * batchSize * sizeof(float), cudaMemcpyDeviceToDevice);

        dim3 blockDim(256);
        dim3 gridDim((outputSize + blockDim.x - 1) / blockDim.x, batchSize);

        forwardKernel << <gridDim, blockDim >> > (input, d_weights, d_bias, d_output, inputSize, outputSize, batchSize);
        cudaDeviceSynchronize();

        // Apply activation function
        float* d_activated_output;
        cudaMalloc(&d_activated_output, outputSize * batchSize * sizeof(float));

        applyActivation(d_output, d_activated_output, outputSize * batchSize, activationType);
        

        // Debugging: Print some values from d_output and d_activated_output for the first batch
        float h_output[10];
        float h_activated_output[10];
        cudaMemcpy(h_output, d_output, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_activated_output, d_activated_output, 10 * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Before activation (first batch):" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << h_output[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "After activation (first batch):" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << h_activated_output[i] << " ";
        }
        std::cout << std::endl;

        cudaFree(d_output);
        d_output = d_activated_output;
        return d_output;
    }

    // Placeholder for backpropagation function
    void backpropagate(const float* gradients, float learningRate) {
        // TODO: Implement backpropagation
        // This will update weights and biases based on the gradients
    }

    int getOutputSize() const { return outputSize; }
    int getBatchSize() const { return batchSize; }
    float* getWeights() const { return d_weights; }
    float* getBias() const { return d_bias; }
};

// Example usage remains the same
float* runNeuralNetwork(float* input, int inputSize, int hiddenSize, int numLayers, int outputSize, int batchSize) {
    std::vector<DenseLayer*> layers;
    // Create layers
    layers.push_back(new DenseLayer(inputSize, hiddenSize, batchSize, "relu"));
    for (int i = 1; i < numLayers - 1; ++i) {
        printf("Creating hidden layer %d\n", i);
        layers.push_back(new DenseLayer(hiddenSize, hiddenSize, batchSize, "relu"));
    }
    layers.push_back(new DenseLayer(hiddenSize, outputSize, batchSize, "softmax"));

    // Forward pass
    printf("numLayers: %d\n", numLayers);
    float* layerInput = input;
    for (int i = 0; i < numLayers; ++i) {
        std::cout << "Layer " << i << " output:" << std::endl;
        layerInput = layers[i]->forward(layerInput);
    }

    // Print the output of the last layer for the first batch
    printf("Output of the last layer (first batch):\n");
    float* output = new float[layers[numLayers - 1]->getOutputSize() * batchSize];
    cudaMemcpy(output, layerInput, layers[numLayers - 1]->getOutputSize() * batchSize * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < layers[numLayers - 1]->getOutputSize(); ++i) {
        printf("%f ", output[i]);
    }
    printf("\n");

	return output;

    // Clean up
    //for (auto& layer : layers) {
    //    delete layer;
    //}
    //delete[] output;
}