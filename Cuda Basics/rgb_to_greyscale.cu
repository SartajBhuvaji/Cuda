#include<cuda.h>
#include<stdio.h>
#include<math.h>

int* read_image(int image_size) {
    int* flattened_image = (int*)malloc(3 * image_size * sizeof(int));

    // Simulate reading an image (fill with dummy data)
    for (int i = 0; i < 3 * image_size; i++) {
        flattened_image[i] = rand() % 256; // Random pixel values
    }
    return flattened_image;
}

__global__ void rgb_to_greyscale(int* rgb_image, int* greyscale_image, int image_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < image_size) {
        int r = rgb_image[3 * idx];     // Red channel
        int g = rgb_image[3 * idx + 1]; // Green channel
        int b = rgb_image[3 * idx + 2]; // Blue channel
        // Convert to greyscale using the luminosity method
        greyscale_image[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

int main() {
    // Step 1: Initialize variables and allocate memory
    int* host_image, *host_greyscale_image;
    int* device_image, *device_greyscale_image;
    int image_size = 1024 * 1024; 
    int image_memory = image_size * sizeof(int); 

    // Step 2: Read the image and prepare the data
    host_image = read_image(image_size);
    host_greyscale_image = (int*)malloc(image_memory); // Greyscale image will have only one channel


    // Step 3: Allocate memory on the device
    cudaMalloc(&device_image, 3 * image_memory);
    cudaMalloc(&device_greyscale_image, image_memory);

    // Step 4: Copy data from host to device
    cudaMemcpy(device_image, host_image, 3 * image_memory, cudaMemcpyHostToDevice);

    // Step 5: Setup execution configuration and launch the kernel
    int no_of_threads = 256;
    int no_of_blocks = (int)ceil((float)image_size / no_of_threads);

    rgb_to_greyscale<<<no_of_blocks, no_of_threads>>>(device_image, device_greyscale_image, image_size);
   
    // Step 6: Copy the result back to the host
    cudaMemcpy(host_greyscale_image, device_greyscale_image, image_memory, cudaMemcpyDeviceToHost);

    // Step 7: Free memory and reset device
    free(host_image);
    free(host_greyscale_image);
    cudaFree(device_image);
    cudaFree(device_greyscale_image);
    cudaDeviceReset(); 
    
    return 0;
}