#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

// CUDA Kernel
__global__ void repeatChars(char *A, int *B, char *STR, int M, int N, int *offsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int offset = offsets[idx];
        char c = A[idx];
        int count = B[idx];
        for (int i = 0; i < count; i++) {
            STR[offset + i] = c;
        }
    }
}

// Host function to compute offset array
void computeOffsets(int *B, int *offsets, int total) {
    offsets[0] = 0;
    for (int i = 1; i < total; i++) {
        offsets[i] = offsets[i - 1] + B[i - 1];
    }
}

int main() {
    int M, N;
    printf("Enter number of rows (M): ");
    scanf("%d", &M);
    printf("Enter number of columns (N): ");
    scanf("%d", &N);

    int total = M * N;

    // Host memory allocation
    char *h_A = (char *)malloc(total * sizeof(char));
    int *h_B = (int *)malloc(total * sizeof(int));
    int *offsets = (int *)malloc(total * sizeof(int));

    printf("Enter character matrix A (%d elements):\n", total);
    for (int i = 0; i < total; i++) {
        scanf(" %c", &h_A[i]); // space before %c to skip whitespace
    }

    printf("Enter integer matrix B (%d elements):\n", total);
    for (int i = 0; i < total; i++) {
        scanf("%d", &h_B[i]);
    }

    // Compute total output string length
    computeOffsets(h_B, offsets, total);
    int strLength = offsets[total - 1] + h_B[total - 1];

    char *h_STR = (char *)malloc((strLength + 1) * sizeof(char)); // +1 for null-terminator

    // Device memory allocation
    char *d_A, *d_STR;
    int *d_B, *d_offsets;

    cudaMalloc((void **)&d_A, total * sizeof(char));
    cudaMalloc((void **)&d_B, total * sizeof(int));
    cudaMalloc((void **)&d_offsets, total * sizeof(int));
    cudaMalloc((void **)&d_STR, strLength * sizeof(char));

    // Copy data to device
    cudaMemcpy(d_A, h_A, total * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, total * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    repeatChars<<<gridSize, blockSize>>>(d_A, d_B, d_STR, M, N, d_offsets);

    // Copy result back to host
    cudaMemcpy(h_STR, d_STR, strLength * sizeof(char), cudaMemcpyDeviceToHost);
    h_STR[strLength] = '\0'; // null-terminate

    printf("\nOutput String STR:\n%s\n", h_STR);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_STR);
    free(offsets);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_offsets);
    cudaFree(d_STR);

    return 0;
}
