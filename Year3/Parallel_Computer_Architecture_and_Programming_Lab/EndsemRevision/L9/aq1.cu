#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Kernel to compute row sums
__global__ void computeRowSums(int *A, int *rowSum, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j];
        }
        rowSum[i] = sum;
    }
}

// Kernel to compute column sums
__global__ void computeColSums(int *A, int *colSum, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        int sum = 0;
        for (int i = 0; i < M; i++) {
            sum += A[i * N + j];
        }
        colSum[j] = sum;
    }
}

// Kernel to compute B[i][j] = rowSum[i] + colSum[j]
__global__ void computeOutput(int *B, int *rowSum, int *colSum, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        B[i * N + j] = rowSum[i] + colSum[j];
    }
}

int main() {
    int M, N;
    printf("Enter number of rows (M): ");
    scanf("%d", &M);
    printf("Enter number of columns (N): ");
    scanf("%d", &N);

    int *A = (int *)malloc(M * N * sizeof(int));
    int *B = (int *)malloc(M * N * sizeof(int));

    printf("Enter the matrix elements:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &A[i * N + j]);
        }
    }

    int *d_A, *d_B, *d_rowSum, *d_colSum;

    cudaMalloc((void **)&d_A, M * N * sizeof(int));
    cudaMalloc((void **)&d_B, M * N * sizeof(int));
    cudaMalloc((void **)&d_rowSum, M * sizeof(int));
    cudaMalloc((void **)&d_colSum, N * sizeof(int));

    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Compute row sums
    computeRowSums<<<(M + 255) / 256, 256>>>(d_A, d_rowSum, M, N);

    // Compute column sums
    computeColSums<<<(N + 255) / 256, 256>>>(d_A, d_colSum, M, N);

    // Compute output matrix B
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);
    computeOutput<<<gridSize, blockSize>>>(d_B, d_rowSum, d_colSum, M, N);

    // Copy result back to host
    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nOutput Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_rowSum);
    cudaFree(d_colSum);

    return 0;
}
