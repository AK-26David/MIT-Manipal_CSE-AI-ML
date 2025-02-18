#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to perform 1D convolution
__global__ void convolution_1d_kernel(float *input, float *mask, float *output, int width, int mask_width) {
    int half_mask = mask_width / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width) {
        float result = 0.0f;

        // Convolve the input array with the mask array
        for (int j = 0; j < mask_width; j++) {
            int input_idx = i + j - half_mask;

            // Check if the input index is within bounds
            if (input_idx >= 0 && input_idx < width) {
                result += input[input_idx] * mask[j];
            }
        }

        // Store the result in the output array
        output[i] = result;
    }
}

void convolution_1d(float *input, float *mask, float *output, int width, int mask_width) {
    // Allocate memory on the device
    float *d_input, *d_mask, *d_output;
    cudaMalloc(&d_input, width * sizeof(float));
    cudaMalloc(&d_mask, mask_width * sizeof(float));
    cudaMalloc(&d_output, width * sizeof(float));

    // Copy input and mask arrays to device memory
    cudaMemcpy(d_input, input, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, mask_width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int block_size = 256;  // Number of threads per block
    int grid_size = (width + block_size - 1) / block_size;  // Number of blocks to cover the array

    // Launch the kernel
    convolution_1d_kernel<<<grid_size, block_size>>>(d_input, d_mask, d_output, width, mask_width);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to host memory
    cudaMemcpy(output, d_output, width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
}

int main() {
    int width = 10;
    int mask_width = 3;

    // Sample input array
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

    // Sample mask array (e.g., a simple averaging filter)
    float mask[] = {0.2f, 0.5f, 0.2f};

    // Output array
    float output[width];

    // Call the CUDA convolution function
    convolution_1d(input, mask, output, width, mask_width);

    // Print the result
    printf("Output after convolution:\n");
    for (int i = 0; i < width; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}
