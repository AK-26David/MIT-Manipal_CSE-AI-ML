#include <stdio.h>
#include <string.h>

#define ROWS 4
#define COLS 100
#define WORD_LEN 20

__device__ bool matchWord(const char* sentence, const char* word, int startIdx, int wordLen) {
    for (int i = 0; i < wordLen; i++) {
        if (sentence[startIdx + i] != word[i])
            return false;
    }
    // Check word boundary
    if ((startIdx + wordLen == COLS) || (sentence[startIdx + wordLen] == ' ' || sentence[startIdx + wordLen] == '\0')) {
        return true;
    }
    return false;
}

__global__ void countWordOccurrences(char input[ROWS][COLS], char* keyword, int keywordLen, int* count) {
    __shared__ char sharedKeyword[WORD_LEN];
    
    // Load keyword into shared memory
    if (threadIdx.x < keywordLen) {
        sharedKeyword[threadIdx.x] = keyword[threadIdx.x];
    }
    __syncthreads();

    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < ROWS && col < COLS) {
        if ((col == 0 || input[row][col - 1] == ' ') &&
            matchWord(&input[row][0], sharedKeyword, col, keywordLen)) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    char h_input[ROWS][COLS] = {
        "the cat and the dog",
        "the bat flew over the cat",
        "another line with the word the",
        "the the the"
    };
    char h_keyword[] = "the";
    int keywordLen = strlen(h_keyword);
    int h_count = 0;

    char(*d_input)[COLS];
    char* d_keyword;
    int* d_count;

    cudaMalloc((void**)&d_input, sizeof(char) * ROWS * COLS);
    cudaMalloc((void**)&d_keyword, sizeof(char) * WORD_LEN);
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy(d_input, h_input, sizeof(char) * ROWS * COLS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keyword, h_keyword, sizeof(char) * keywordLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    countWordOccurrences<<<ROWS, COLS>>>(d_input, d_keyword, keywordLen, d_count);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word \"%s\" appears %d times.\n", h_keyword, h_count);

    cudaFree(d_input);
    cudaFree(d_keyword);
    cudaFree(d_count);

    return 0;
}