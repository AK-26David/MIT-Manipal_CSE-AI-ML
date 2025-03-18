#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Matrix dimensions (You can modify these as per your needs)
#define M 3  // Rows of matrices
#define N 3  // Columns of matrices

// Kernel to add matrices - Approach (a): Each row of resultant matrix computed by one thread
__global__ void addMatricesRowWise(int *A, int *B, int *C, int numCols) {
    int row = blockIdx.x;  // Thread index corresponds to the row
    if (row < M) {
        for (int col = 0; col < numCols; col++) {
            C[row * numCols + col] = A[row * numCols + col] + B[row * numCols + col];
        }
    }
}

// Kernel to add matrices - Approach (b): Each column of resultant matrix computed by one thread
__global__ void addMatricesColWise(int *A, int *B, int *C, int numRows) {
    int col = blockIdx.x;  // Thread index corresponds to the column
    if (col < N) {
        for (int row = 0; row < numRows; row++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

// Kernel to add matrices - Approach (c): Each element of resultant matrix computed by one thread
__global__ void addMatricesElementWise(int *A, int *B, int *C) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
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

    size_t size = M * N * sizeof(int);
    
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = i - j;
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // --- Approach (a): Each row of the resultant matrix computed by one thread ---
    addMatricesRowWise<<<M, 1>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Resultant Matrix (Approach a - Row-wise):\n");
    printMatrix(C, M, N);

    // --- Approach (b): Each column of the resultant matrix computed by one thread ---
    addMatricesColWise<<<N, 1>>>(d_A, d_B, d_C, M);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nResultant Matrix (Approach b - Column-wise):\n");
    printMatrix(C, M, N);

    // --- Approach (c): Each element of the resultant matrix computed by one thread ---
    addMatricesElementWise<<<M, N>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
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
