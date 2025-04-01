#include <stdio.h>
#include <cuda.h>
#include <cstdlib>

#define N 16           // Smaller input array size for easy verification
#define KERNEL_SIZE 5   // Smaller kernel size
#define BLOCK_SIZE 16

// Constant memory for the kernel
__constant__ float d_kernel[KERNEL_SIZE];

__global__ void convolution1D(const float *d_input, float *d_output, int data_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = kernel_size / 2;

    if (idx < data_size) {
        float sum = 0.0f;

        // Apply convolution using constant memory
        for (int i = -radius; i <= radius; i++) {
            int neighbor_index = idx + i;

            if (neighbor_index >= 0 && neighbor_index < data_size) {
                sum += d_input[neighbor_index] * d_kernel[i + radius];
            }
        }
        d_output[idx] = sum;
    }
}

void initializeData(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (rand() % 100) / 10.0f;  // Random float values between 0.0 and 9.9
    }
}

void printArray(const float *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%0.2f ", array[i]);
        if ((i + 1) % 8 == 0) printf("\n");  // Print 8 elements per line for readability
    }
    printf("\n");
}

int main() {
    size_t bytes_data = N * sizeof(float);
    size_t bytes_kernel = KERNEL_SIZE * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(bytes_data);
    float *h_output = (float *)malloc(bytes_data);
    float h_kernel[KERNEL_SIZE] = {0.1, 0.2, 0.4, 0.2, 0.1};  // Example kernel

    // Initialize input data
    initializeData(h_input, N);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, bytes_data);
    cudaMalloc((void **)&d_output, bytes_data);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, bytes_data, cudaMemcpyHostToDevice);

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, h_kernel, bytes_kernel);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    convolution1D<<<gridDim, blockDim>>>(d_input, d_output, N, KERNEL_SIZE);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes_data, cudaMemcpyDeviceToHost);

    // Print results
    printf("\nInput Array:\n");
    printArray(h_input, N);

    printf("\nKernel:\n");
    printArray(h_kernel, KERNEL_SIZE);

    printf("\nOutput Array:\n");
    printArray(h_output, N);

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
