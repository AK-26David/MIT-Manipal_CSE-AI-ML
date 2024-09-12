#include <stdio.h>
#include <string.h>

#define MAX_CHARS 256

// Function to compute the bad character shift table for Boyer-Moore algorithm
void computeBadCharTable(char* pattern, int patternLength, int badCharTable[]) {
    // Initialize bad character table with default shift value
    for (int i = 0; i < MAX_CHARS; i++) {
        badCharTable[i] = patternLength;
    }

    // Compute shift values for characters in the pattern
    for (int i = 0; i < patternLength - 1; i++) {
        badCharTable[(int)pattern[i]] = patternLength - 1 - i;
    }
}

// Function to compute the suffix shift table for Boyer-Moore algorithm
void computeSuffixTable(char* pattern, int patternLength, int suffixTable[]) {
    // Initialize suffix table
    int lastPrefixPosition = patternLength;

    // Case 1: No prefix
    for (int i = patternLength - 1; i >= 0; i--) {
        if (isSuffix(pattern, patternLength, i + 1)) {
            lastPrefixPosition = i + 1;
        }
        suffixTable[i] = lastPrefixPosition - i + patternLength - 1;
    }

    // Case 2: Non-empty prefix
    for (int i = 0; i < patternLength - 1; i++) {
        int length = longestPrefix(pattern, patternLength, i);
        suffixTable[length] = patternLength - 1 - i + length;
    }
}

// Function to check if a substring of the pattern is a suffix
int isSuffix(char* pattern, int patternLength, int pos) {
    int j = 0;
    for (int i = pos; i < patternLength; i++) {
        if (pattern[i] != pattern[j]) {
            return 0;
        }
        j++;
    }
    return 1;
}

// Function to compute the length of the longest prefix ending at pos
int longestPrefix(char* pattern, int patternLength, int pos) {
    int length = 0;
    int i = pos;
    int j = patternLength - 1;

    while (i >= 0 && pattern[i] == pattern[j]) {
        length++;
        i--;
        j--;
    }

    return length;
}

// Function to perform string matching using Boyer-Moore algorithm
int boyerMooreSearch(char* text, char* pattern, int* comparisons) {
    int textLength = strlen(text);
    int patternLength = strlen(pattern);

    // Step 1: Compute the bad character and suffix tables
    int badCharTable[MAX_CHARS];
    int suffixTable[patternLength];
    computeBadCharTable(pattern, patternLength, badCharTable);
    computeSuffixTable(pattern, patternLength, suffixTable);

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

        // Apply bad character and good suffix rules to find the next alignment
        int badCharShift = badCharTable[(int)text[i - k]];
        int suffixShift = suffixTable[k];

        i += (badCharShift > suffixShift) ? badCharShift : suffixShift;
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

    int index = boyerMooreSearch(text, pattern, &comparisons_successful);

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

    int unsuccessfulIndex = boyerMooreSearch(text, anotherPattern, &comparisons_unsuccessful);

    if (unsuccessfulIndex == -1) {
        printf("Another pattern not found in the text\n");
        printf("Number of comparisons in unsuccessful search: %d\n", comparisons_unsuccessful);
    }

    return 0;
}
