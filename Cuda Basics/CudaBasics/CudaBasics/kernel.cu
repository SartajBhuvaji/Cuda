// Linear regression using gradient descent in CUDA

#include <cuda.h>
#include <stdio.h>
#include <vector>

__global__ void gradient_descent(const float* __restrict__ X, const float* __restrict__ Y, float* theta, const float alpha, const int m) {
	__shared__ float s_gradient[2]; // Shared memory for gradient
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Initialize shared memory
    if (tid < 2) {
		s_gradient[tid] = 0.0f; // Initialize shared memory to 0
    }
    __syncthreads();

    // Compute partial sums
	float local_gradient[2] = { 0.0f, 0.0f }; // Local gradient for each thread
    for (int i = tid; i < m; i += stride) {
        const float prediction = theta[0] + theta[1] * X[i];
        const float error = prediction - Y[i];
        local_gradient[0] += error;
        local_gradient[1] += error * X[i];
    }

    // Reduce within the block
    for (int j = 0; j < 2; ++j) { 
        atomicAdd(&s_gradient[j], local_gradient[j]);
    }
	__syncthreads(); // Wait for all threads to finish updating shared memory

    // Update theta
    if (tid < 2) {
		float gradient = s_gradient[tid] / m; // Average gradient
		theta[tid] -= alpha * gradient; // Update theta
    }
}


std::vector<std::vector<float>> get_data() {
    // Sample dataset
    std::vector<std::vector<float>> data = {
        {1.0, 2.0},   // {X, Y}
        {2.0, 3.5},
        {3.0, 7.0},   // Outlier
        {4.0, 6.5},
        {5.0, 8.0},
        {6.0, 11.0},  // Outlier
        {7.0, 9.5},
        {8.0, 11.5},
        {9.0, 10.0},  // Outlier
        {10.0, 13.0},
        {11.0, 14.5},
        {12.0, 16.0},
        {13.0, 19.0}, // Outlier
        {14.0, 17.5},
        {15.0, 20.0}
    };
    return data;
}

void initialize_data(const std::vector<std::vector<float>>& data, float*& d_X, float*& d_Y, int& m) {
    m = data.size();
    float* h_X = new float[m];
    float* h_Y = new float[m];

    for (int i = 0; i < m; ++i) {
        h_X[i] = data[i][0];
        h_Y[i] = data[i][1];
    }

    cudaMalloc(&d_X, m * sizeof(float));
    cudaMalloc(&d_Y, m * sizeof(float));
    cudaMemcpy(d_X, h_X, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, m * sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_X;
    delete[] h_Y;
}

int main() {

	// Load data
	std::vector<std::vector<float>> data = get_data();

    float* d_X, * d_Y, * d_theta;
	int m; // size of dataset

	// Initialize data
    initialize_data(data, d_X, d_Y, m);

    // Allocate memory for theta (parameters)
    float h_theta[2] = { 0.0f, 0.0f }; // Initialize theta to 0 (intercept and slope)
    cudaMalloc(&d_theta, 2 * sizeof(float));
    cudaMemcpy(d_theta, h_theta, 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Set learning rate and number of iterations
	float alpha = 0.01; // Learning rate
    int num_iters = 1000;
    int threadsPerBlock = 256; 
    int blocksPerGrid = 1; 

    for (int iter = 0; iter < num_iters; ++iter) {
        gradient_descent << <blocksPerGrid, threadsPerBlock >> > (d_X, d_Y, d_theta, alpha, m);
        cudaDeviceSynchronize(); // Ensure kernel execution is finished before the next iteration
    }

    // Copy results back to host
    cudaMemcpy(h_theta, d_theta, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the best fit line parameters (intercept and slope)
    printf("Best fit line parameters:\n");
    printf("Intercept (Theta[0]): %f\n", h_theta[0]);
    printf("Slope (Theta[1]): %f\n", h_theta[1]);

    // Free memory
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_theta);

    return 0;
}