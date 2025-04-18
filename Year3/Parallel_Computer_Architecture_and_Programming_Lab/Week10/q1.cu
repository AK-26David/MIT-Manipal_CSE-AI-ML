#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_SIZE 16  // Block size (16x16)

__global__ void matrixMul(const float *A, const float *B, float *C, int N) {
    // Row and column indices for C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMultiplicationHost(float *A, float *B, float *C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy the result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void printMatrix(float *M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", M[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int N = 4;  // Matrix size (N x N)
    size_t size = N * N * sizeof(float);

    // Allocate and initialize matrices
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    // Initialize matrices with sample values
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(rand() % 10);
        B[i] = (float)(rand() % 10);
    }

    printf("\nMatrix A:\n");
    printMatrix(A, N);

    printf("\nMatrix B:\n");
    printMatrix(B, N);

    // Perform matrix multiplication
    matrixMultiplicationHost(A, B, C, N);

    printf("\nMatrix C (Result):\n");
    printMatrix(C, N);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
