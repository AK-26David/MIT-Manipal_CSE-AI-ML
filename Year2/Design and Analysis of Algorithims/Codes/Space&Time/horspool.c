#include <stdio.h>
#include <string.h>

#define MAX_CHARS 256

// Function to compute the shift table for Horspool's algorithm
void computeShiftTable(char* pattern, int patternLength, int shiftTable[]) {
    // Initialize shift table with pattern length
    for (int i = 0; i < MAX_CHARS; i++) {
        shiftTable[i] = patternLength;
    }

    // Compute the shift values for characters in the pattern
    for (int i = 0; i < patternLength - 1; i++) {
        shiftTable[(int)pattern[i]] = patternLength - 1 - i;
    }
}

// Function to perform string matching using Horspool's algorithm
int horspoolSearch(char* text, char* pattern, int* comparisons) {
    int textLength = strlen(text);
    int patternLength = strlen(pattern);

    // Step 1: Compute the shift table for the pattern
    int shiftTable[MAX_CHARS];
    computeShiftTable(pattern, patternLength, shiftTable);

    // Step 2: Perform string matching
    int i = patternLength - 1;
    while (i < textLength) {
        int k = 0;
        // Compare characters from right to left
        while (k < patternLength && pattern[patternLength - 1 - k] == text[i - k]) {
            k++;
            (*comparisons)++;
        }

        if (k == patternLength) {
            // Match found at index i - patternLength + 1
            return i - patternLength + 1;
        }

        // Update i using the shift table
        i += shiftTable[(int)text[i]];
    }

    return -1; // Match not found
}

int main() {
    char text[100], pattern[100];
    int comparisons_successful = 0;
    int comparisons_unsuccessful = 0;

    printf("Enter the text: ");
    scanf("%[^\n]%*c", text);

    printf("Enter the pattern to search: ");
    scanf("%[^\n]%*c", pattern);

    int index = horspoolSearch(text, pattern, &comparisons_successful);

    if (index != -1) {
        printf("Pattern found at index: %d\n", index);
        printf("Number of comparisons in successful search: %d\n", comparisons_successful);
    } else {
        printf("Pattern not found in the text\n");
        printf("Number of comparisons in successful search: %d\n", comparisons_successful);
    }

    // Perform an unsuccessful search with a different pattern
    char anotherPattern[100];
    printf("Enter another pattern for an unsuccessful search: ");
    scanf("%[^\n]%*c", anotherPattern);

    int unsuccessfulIndex = horspoolSearch(text, anotherPattern, &comparisons_unsuccessful);

    if (unsuccessfulIndex == -1) {
        printf("Another pattern not found in the text\n");
        printf("Number of comparisons in unsuccessful search: %d\n", comparisons_unsuccessful);
    }

    return 0;
}
