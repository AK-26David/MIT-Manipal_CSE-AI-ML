#include <stdio.h>
#include <cuda_runtime.h>

__global__ void transformMatrix(int *A, int *B, int *rowSums, int *colSums, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        int val = A[row * n + col];
        if (val % 2 == 0)
            B[row * n + col] = rowSums[row];
        else
            B[row * n + col] = colSums[col];
    }
}

int main()
{
    int m, n;
    printf("Enter number of rows (M): ");
    scanf("%d", &m);
    printf("Enter number of columns (N): ");
    scanf("%d", &n);

    int A[100][100], B[100][100];
    int rowSums[100] = {0}, colSums[100] = {0};

    printf("Enter elements of %dx%d matrix A:\n", m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            scanf("%d", &A[i][j]);
            rowSums[i] += A[i][j];
            colSums[j] += A[i][j];
        }

    // Device pointers
    int *d_A, *d_B, *d_rowSums, *d_colSums;

    cudaMalloc((void **)&d_A, m * n * sizeof(int));
    cudaMalloc((void **)&d_B, m * n * sizeof(int));
    cudaMalloc((void **)&d_rowSums, m * sizeof(int));
    cudaMalloc((void **)&d_colSums, n * sizeof(int));

    cudaMemcpy(d_A, A, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowSums, rowSums, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colSums, colSums, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + 15) / 16, (m + 15) / 16);

    transformMatrix<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_rowSums, d_colSums, m, n);

    cudaMemcpy(B, d_B, m * n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nMatrix B (Result):\n");
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
            printf("%d ", B[i][j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_rowSums);
    cudaFree(d_colSums);

    return 0;
}