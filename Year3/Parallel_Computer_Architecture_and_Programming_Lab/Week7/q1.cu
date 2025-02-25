#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// Define maximum number of words in the sentence and maximum word length
#define MAX_WORDS 1000
#define MAX_WORD_LENGTH 100

__global__ void countWordKernel(char *sentence, char *word, int *wordCount, int sentenceLength, int wordLength) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread processes one word in the sentence
    if (index < sentenceLength) {
        int wordIndex = 0;
        int match = 1;

        // Check if the word from sentence matches the target word
        for (int j = 0; j < wordLength; j++) {
            if (sentence[index + j] != word[j]) {
                match = 0;
                break;
            }
        }

        // If there is a match, increment the counter atomically
        if (match == 1) {
            atomicAdd(wordCount, 1);
        }
    }
}

int main() {
    // Input sentence and target word
    char sentence[] = "this is a test sentence with test words, testing is fun test test test";
    char word[] = "test";

    int sentenceLength = strlen(sentence);
    int wordLength = strlen(word);

    // Allocate memory for word count
    int *wordCount;
    cudaMallocManaged(&wordCount, sizeof(int));
    *wordCount = 0;

    // Prepare sentence and word on the GPU
    char *d_sentence, *d_word;
    cudaMallocManaged(&d_sentence, sentenceLength + 1);  // +1 for null-terminator
    cudaMallocManaged(&d_word, wordLength + 1);          // +1 for null-terminator

    cudaMemcpy(d_sentence, sentence, sentenceLength + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, wordLength + 1, cudaMemcpyHostToDevice);

    // Define number of threads per block and number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (sentenceLength + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    countWordKernel<<<blocksPerGrid, threadsPerBlock>>>(d_sentence, d_word, wordCount, sentenceLength, wordLength);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Output the result
    printf("The word '%s' appears %d times in the sentence.\n", word, *wordCount);

    // Clean up
    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(wordCount);

    return 0;
}
