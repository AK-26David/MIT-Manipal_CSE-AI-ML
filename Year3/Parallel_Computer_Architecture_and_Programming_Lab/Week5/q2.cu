#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256


__global__ void addVectors(int* A, int* B, int* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024; 
    int *A, *B, *C; 
    int *d_A, *d_B, *d_C; 


    A = (int*)malloc(N * sizeof(int));
    B = (int*)malloc(N * sizeof(int));
    C = (int*)malloc(N * sizeof(int));


    for (int i = 0; i < N; i++) {
        A[i] = i + 1; // Vector A: 1, 2, 3, 4, ...
        B[i] = (i + 1) * 2; // Vector B: 2, 4, 6, 8, ...
    }

  
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

   
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);


    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

 
    addVectors<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);


    cudaDeviceSynchronize();

   
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");


    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
