#include <stdio.h>

#define MAX_CITIES 10

int adjacencyMatrix[MAX_CITIES][MAX_CITIES];

// Function to calculate the total distance of a given path
int calculateTotalDistance(int path[], int numCities) {
    int totalDistance = 0;
    for (int i = 0; i < numCities - 1; i++) {
        totalDistance += adjacencyMatrix[path[i]][path[i + 1]];
    }
    // Add the distance from the last city back to the starting city
    totalDistance += adjacencyMatrix[path[numCities - 1]][path[0]];
    return totalDistance;
}

// Function to swap elements at two indices in an array
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}
// Function to generate all permutations of a given array
void generatePermutations(int path[], int start, int numCities, int* minDistance, int* optimalPath) {
    if (start == numCities - 1) {
        // Calculate the total distance for the current permutation
        int currentDistance = calculateTotalDistance(path, numCities);       
        // Update the minimum distance and optimal path if needed
        if (currentDistance < *minDistance) {
            *minDistance = currentDistance;
            for (int i = 0; i < numCities; i++) {
                optimalPath[i] = path[i];
            }
        }
    } else {
        for (int i = start; i < numCities; i++) {
            // Swap elements to generate different permutations
            swap(&path[start], &path[i]);
            
            // Recursively generate permutations
            generatePermutations(path, start + 1, numCities, minDistance, optimalPath);
            
            // Swap back to the original order for backtracking
            swap(&path[start], &path[i]);
        }
    }
}

int main() {
    int numCities;

    printf("Enter the number of cities: ");
    scanf("%d", &numCities);

    // Input the adjacency matrix
    printf("Enter the adjacency matrix:\n");
    for (int i = 0; i < numCities; i++) {
        for (int j = 0; j < numCities; j++) {
            scanf("%d", &adjacencyMatrix[i][j]);
        }
    }

    int path[MAX_CITIES];
    int minDistance = 1 << 30;  // A large initial value
    int optimalPath[MAX_CITIES];

    // Initialize the path array with city indices
    for (int i = 0; i < numCities; i++) {
        path[i] = i;
    }

    // Generate all permutations and find the optimal path
    generatePermutations(path, 0, numCities, &minDistance, optimalPath);

    // Output the result
    printf("Optimal Path: ");
    for (int i = 0; i < numCities; i++) {
        printf("%d ", optimalPath[i] + 1);  // Adding 1 to convert to 1-based indexing
    }
    printf("\nMinimum Distance: %d\n", minDistance);

    return 0;
}
