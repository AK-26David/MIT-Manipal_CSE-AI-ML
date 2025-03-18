#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Matrix dimensions (you can change these values as needed)
#define M 3  // Rows of Matrix A and Resultant Matrix C
#define N 3  // Columns of Matrix B and Resultant Matrix C
#define K 3  // Columns of Matrix A and Rows of Matrix B

// Kernel to multiply matrices - Approach (a): Each row of resultant matrix computed by one thread
__global__ void multiplyMatricesRowWise(int *A, int *B, int *C, int numColsA, int numColsB) {
    int row = blockIdx.x;  // Each thread computes one row of the resultant matrix
    if (row < M) {
        for (int col = 0; col < numColsB; col++) {
            int sum = 0;
            for (int k = 0; k < numColsA; k++) {
                sum += A[row * numColsA + k] * B[k * numColsB + col];
            }
            C[row * numColsB + col] = sum;
        }
    }
}

// Kernel to multiply matrices - Approach (b): Each column of resultant matrix computed by one thread
__global__ void multiplyMatricesColWise(int *A, int *B, int *C, int numRowsA, int numColsB) {
    int col = blockIdx.x;  // Each thread computes one column of the resultant matrix
    if (col < N) {
        for (int row = 0; row < numRowsA; row++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// Kernel to multiply matrices - Approach (c): Each element of resultant matrix computed by one thread
__global__ void multiplyMatricesElementWise(int *A, int *B, int *C, int numColsA, int numColsB) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * numColsA + k] * B[k * numColsB + col];
        }
        C[row * numColsB + col] = sum;
    }
}

// Helper function to print matrix
void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Main function
int main() {
    // Allocate and initialize matrices
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    size_t sizeA = M * K * sizeof(int);
    size_t sizeB = K * N * sizeof(int);
    size_t sizeC = M * N * sizeof(int);
    
    A = (int*)malloc(sizeA);
    B = (int*)malloc(sizeB);
    C = (int*)malloc(sizeC);

    // Initialize matrices A and B with some values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = i + j; // Simple values for illustration
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = i - j; // Simple values for illustration
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Copy data to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // --- Approach (a): Each row of the resultant matrix computed by one thread ---
    multiplyMatricesRowWise<<<M, 1>>>(d_A, d_B, d_C, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("Resultant Matrix (Approach a - Row-wise):\n");
    printMatrix(C, M, N);

    // --- Approach (b): Each column of the resultant matrix computed by one thread ---
    multiplyMatricesColWise<<<N, 1>>>(d_A, d_B, d_C, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("\nResultant Matrix (Approach b - Column-wise):\n");
    printMatrix(C, M, N);

    // --- Approach (c): Each element of the resultant matrix computed by one thread ---
    multiplyMatricesElementWise<<<M, N>>>(d_A, d_B, d_C, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("\nResultant Matrix (Approach c - Element-wise):\n");
    printMatrix(C, M, N);

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
