#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel to perform Odd-Even Transposition Sort
__global__ void odd_even_transposition_sort(int *arr, int n, bool phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n - 1) {
        // Odd phase: Compare arr[i] and arr[i+1] where i is odd
        if (phase) {
            if (idx % 2 == 1 && arr[idx] > arr[idx + 1]) {
                // Swap elements
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
        // Even phase: Compare arr[i] and arr[i+1] where i is even
        else {
            if (idx % 2 == 0 && arr[idx] > arr[idx + 1]) {
                // Swap elements
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
    }
}

// Function to perform Odd-Even Transposition Sort using CUDA
void odd_even_sort(int *arr, int n) {
    // Allocate device memory
    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));

    // Copy input array to device
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int block_size = 256;  // Number of threads per block
    int grid_size = (n + block_size - 1) / block_size;  // Number of blocks to cover the array

    // Run the odd-even transposition sort in multiple iterations
    for (int phase = 0; phase < n; phase++) {
        // Launch kernel for the current phase (odd or even)
        odd_even_transposition_sort<<<grid_size, block_size>>>(d_arr, n, phase % 2);
        
        // Wait for kernel to finish
        cudaDeviceSynchronize();
    }

    // Copy the sorted array back to host
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);
}

int main() {
    int arr[] = {29, 10, 14, 37, 13, 35, 55, 22, 90, 2};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // Call the Odd-Even Transposition Sort function
    odd_even_sort(arr, n);

    // Print the sorted array
    printf("Sorted Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
