#include <stdio.h>

#define MAX_SIZE 10

int costMatrix[MAX_SIZE][MAX_SIZE];
int assignment[MAX_SIZE];
int minCost = 1 << 30;  // A large initial value

void calculateCost(int n) {
    int currentCost = 0;

    for (int i = 0; i < n; i++) {
        currentCost += costMatrix[i][assignment[i]];
    }

    if (currentCost < minCost) {
        minCost = currentCost;
    }
}

void generatePermutations(int n, int currentIndex) {
    if (currentIndex == n) {
        calculateCost(n);
        return;
    }

    for (int i = currentIndex; i < n; i++) {
        // Swap elements to generate different permutations
        int temp = assignment[currentIndex];
        assignment[currentIndex] = assignment[i];
        assignment[i] = temp;

        generatePermutations(n, currentIndex + 1);

        // Swap back to the original order for backtracking
        temp = assignment[currentIndex];
        assignment[currentIndex] = assignment[i];
        assignment[i] = temp;
    }
}

int main() {
    int n;

    printf("Enter the number of workers/jobs: ");
    scanf("%d", &n);

    printf("Enter the cost matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &costMatrix[i][j]);
        }
    }

    for (int i = 0; i < n; i++) {
        assignment[i] = i;  // Initial assignment
    }

    generatePermutations(n, 0);

    printf("Minimum cost of assignment: %d\n", minCost);

    return 0;
}
