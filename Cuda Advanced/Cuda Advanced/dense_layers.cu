#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
//#include<C:\\Users\\sbhuv\\Desktop\\Cuda\\Cuda\\Cuda Advanced\\Cuda Advanced\\activations.cu>

__global__ void forwardKernel(const float* input, const float* weights, const float* bias,
    float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[i * outputSize + idx];
        }
        output[idx] = sum + bias[idx];
    }
}

class DenseLayer {
private:
    int inputSize;
    int outputSize;
    float* d_weights;
    float* d_bias;
    float* d_output;
    float* d_input;  // Store input for backpropagation
    const char* activationType;

    void initializeWeightsAndBiases() {
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

        float std_dev = sqrt(2.0f / inputSize);  // He initialization

        curandGenerateNormal(gen, d_weights, inputSize * outputSize, 0, std_dev);
        curandGenerateNormal(gen, d_bias, outputSize, 0, std_dev);

        curandDestroyGenerator(gen);
    }

public:
    DenseLayer(int inSize, int outSize, const char* actType)
        : inputSize(inSize), outputSize(outSize), activationType(actType) {
        cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_bias, outputSize * sizeof(float));
        cudaMalloc(&d_output, outputSize * sizeof(float));
        cudaMalloc(&d_input, inputSize * sizeof(float));

        initializeWeightsAndBiases();
    }

    ~DenseLayer() {
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_output);
        cudaFree(d_input);
    }

  

    float* forward(const float* input) {
        cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyDeviceToDevice);

        int blockSize = 256;
        int numBlocks = (outputSize + blockSize - 1) / blockSize;

        forwardKernel << <numBlocks, blockSize >> > (input, d_weights, d_bias, d_output, inputSize, outputSize);

        // Apply activation function
        float* d_activated_output;
        cudaMalloc(&d_activated_output, outputSize * sizeof(float));
        applyActivation(d_output, d_activated_output, outputSize, activationType);

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
    float* getWeights() const { return d_weights; }
    float* getBias() const { return d_bias; }
};

//// Example usage
void runNeuralNetwork(float* input, int inputSize, int hiddenSize, int outputSize, int numLayers) {
    DenseLayer* layers[3];

    // Create layers
    layers[0] = new DenseLayer(inputSize, hiddenSize, "relu");
    for (int i = 1; i < numLayers - 1; ++i) {
        layers[i] = new DenseLayer(hiddenSize, hiddenSize, "relu");
    }
    layers[numLayers - 1] = new DenseLayer(hiddenSize, outputSize, "softmax");

    printf("Layers created\n");

     //Forward pass
    /*float* layerInput = input;
    for (int i = 0; i < numLayers; ++i) {
        layerInput = layers[i]->forward(layerInput);
    }*/

     //   // The output of the last layer is now in layerInput
     //   
        //// Backpropagation

     //   // Print output after last later
        //float* output = new float[layers[numLayers - 1]->getOutputSize()];
        //cudaMemcpy(output, layerInput, layers[numLayers - 1]->getOutputSize() * sizeof(float), cudaMemcpyDeviceToHost);

        //printf("Output: ");
        //for (int i = 0; i < layers[numLayers - 1]->getOutputSize(); ++i) {
        //	printf("%f ", output[i]);
        //}

     //   // TODO: Implement backpropagation

     //   // Clean up
     //   for (int i = 0; i < numLayers; ++i) {
     //       delete layers[i];
       //}

}