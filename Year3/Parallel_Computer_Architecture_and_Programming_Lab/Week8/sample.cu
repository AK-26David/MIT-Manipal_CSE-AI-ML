#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Size of the matrix (NxN)

__global__ void matrixTransposeKernel(int* d_input, int* d_output, int width) {
    // Calculate the global thread index for a 2D block
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        // Transpose the element
        d_output[col * width + row] = d_input[row * width + col];
    }
}

void matrixTranspose(int* h_input, int* h_output, int width) {
    int size = width * width * sizeof(int);

    int *d_input, *d_output;

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy the input matrix to the device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 threadsPerBlock(16, 16);  // Thread block size (16x16 threads per block)
    dim3 numBlocks((width + 15) / 16, (width + 15) / 16);  // Grid size (ensuring full coverage)

    // Launch the kernel to compute the transpose
    matrixTransposeKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the transposed matrix back to the host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int width = N;  // Set the matrix dimension (NxN)
    int* h_input = (int*)malloc(width * width * sizeof(int));
    int* h_output = (int*)malloc(width * width * sizeof(int));

    // Initialize the input matrix with some values
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = i * width + j;  // Fill with some pattern
        }
    }

    // Call matrixTranspose to compute the transpose in parallel using CUDA
    matrixTranspose(h_input, h_output, width);

    // Print the transposed matrix (for small sizes, for debugging purposes)
    printf("Original Matrix (Input):\n");
    for (int i = 0; i < 10 && i < width; i++) {
        for (int j = 0; j < 10 && j < width; j++) {
            printf("%d ", h_input[i * width + j]);
        }
        printf("\n");
    }

    printf("\nTransposed Matrix (Output):\n");
    for (int i = 0; i < 10 && i < width; i++) {
        for (int j = 0; j < 10 && j < width; j++) {
            printf("%d ", h_output[i * width + j]);
        }
        printf("\n");
    }

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
