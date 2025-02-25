#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void replicateStringKernel(char *d_output, const char *d_input, int input_len, int repeat_count, int output_len) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within bounds
    if (idx < output_len) {
        // Calculate which character in input string S the current thread should copy
        int input_index = idx % input_len;

        // Copy the character from the input string to the output string
        d_output[idx] = d_input[input_index];
    }
}

int main() {
    // Input string S
    const char *input_str = "PCAP";
    int input_len = strlen(input_str);
    int repeat_count = 3;  // We want to repeat the string 3 times

    // Total length of the output string RS
    int output_len = input_len * repeat_count;

    // Allocate memory for input and output strings on the host
    char *h_input = (char *)malloc(input_len + 1);  // +1 for null terminator
    char *h_output = (char *)malloc(output_len + 1); // +1 for null terminator

    // Copy the input string to host array
    strcpy(h_input, input_str);

    // Allocate memory on the device
    char *d_input, *d_output;
    cudaMalloc((void**)&d_input, input_len * sizeof(char));
    cudaMalloc((void**)&d_output, output_len * sizeof(char));

    // Copy input string from host to device
    cudaMemcpy(d_input, h_input, input_len * sizeof(char), cudaMemcpyHostToDevice);

    // Launch the kernel with a sufficient number of blocks and threads
    int threads_per_block = 256;
    int blocks_per_grid = (output_len + threads_per_block - 1) / threads_per_block;
    replicateStringKernel<<<blocks_per_grid, threads_per_block>>>(d_output, d_input, input_len, repeat_count, output_len);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, output_len * sizeof(char), cudaMemcpyDeviceToHost);

    // Add null terminator to the output string
    h_output[output_len] = '\0';

    // Output the result
    printf("Input string: %s\n", input_str);
    printf("Output string: %s\n", h_output);

    // Free the device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free the host memory
    free(h_input);
    free(h_output);

    return 0;
}
