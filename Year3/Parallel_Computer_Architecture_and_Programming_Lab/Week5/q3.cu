#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void computeSine(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  
    if (idx < N) {
        output[idx] = sin(input[idx]);  
    }
}

int main() {
    int N = 5;  
    
    float *h_input, *h_output;

    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    h_input[0] = 0.0f;               
    h_input[1] = M_PI / 2.0f;         
    h_input[2] = M_PI;               
    h_input[3] = 3.0f * M_PI / 2.0f;  
    h_input[4] = 2.0f * M_PI;        


    float *d_input, *d_output;
    
   
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));


    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    

    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;


    computeSine<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N);


    cudaDeviceSynchronize();


    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

   
    printf("Sine values of the angles in radians:\n");
    for (int i = 0; i < N; i++) {
        printf("sin(%.2f) = %.4f\n", h_input[i], h_output[i]);
    }

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
