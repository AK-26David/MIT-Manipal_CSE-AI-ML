#include <stdio.h>
#include <cuda.h>

#define TILE_SIZE 16
#define MASK_WIDTH 5
#define RADIUS (MASK_WIDTH / 2)

__constant__ float d_mask[MASK_WIDTH];

// CUDA kernel using shared memory tiling
__global__ void tiled1DConvolution(float *d_input, float *d_output, int width) {
    __shared__ float tile[TILE_SIZE + 2 * RADIUS];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int tile_start = blockIdx.x * blockDim.x;

    // Load elements into shared memory (with halo on both sides)
    int halo_left = global_idx - RADIUS;
    if (halo_left >= 0)
        tile[tid] = d_input[halo_left];
    else
        tile[tid] = 0.0f;

    int halo_right = global_idx + RADIUS;
    if (tid >= blockDim.x - RADIUS) {
        if (halo_right < width)
            tile[tid + 2 * RADIUS] = d_input[halo_right];
        else
            tile[tid + 2 * RADIUS] = 0.0f;
    }

    // Center part
    if (tid < RADIUS)
        tile[tid + blockDim.x] = (global_idx < width) ? d_input[global_idx + RADIUS] : 0.0f;

    __syncthreads();

    // Perform convolution
    float result = 0.0f;
    if (global_idx < width) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            result += tile[tid + j] * d_mask[j];
        }
        d_output[global_idx] = result;
    }
}

int main() {
    int width = 32;
    size_t size = width * sizeof(float);

    // Host arrays
    float h_input[width], h_output[width], h_mask[MASK_WIDTH];

    // Initialize input and mask
    for (int i = 0; i < width; i++) h_input[i] = i + 1;
    for (int i = 0; i < MASK_WIDTH; i++) h_mask[i] = 1.0f / MASK_WIDTH;  // simple average filter

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask, MASK_WIDTH * sizeof(float));

    // Launch kernel
    int blockSize = TILE_SIZE;
    int gridSize = (width + blockSize - 1) / blockSize;
    tiled1DConvolution<<<gridSize, blockSize>>>(d_input, d_output, width);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print result
    printf("Input:\n");
    for (int i = 0; i < width; i++) printf("%0.1f ", h_input[i]);
    printf("\n\nConvolved Output:\n");
    for (int i = 0; i < width; i++) printf("%0.2f ", h_output[i]);
    printf("\n");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
