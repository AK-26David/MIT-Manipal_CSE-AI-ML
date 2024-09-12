#include <stdio.h>

#define MAX_ITEMS 10

typedef struct {
    int weight;
    int value;
} Item;

int knapsackCapacity;
Item items[MAX_ITEMS];
int selectedItems[MAX_ITEMS];
int maxTotalValue = 0;

void generateCombinations(int n, int currentIndex, int currentWeight, int currentValue) {
    if (currentWeight <= knapsackCapacity && currentValue > maxTotalValue) {
        maxTotalValue = currentValue;

        // Save the current selection of items
        for (int i = 0; i < n; i++) {
            selectedItems[i] = (currentIndex >> i) & 1;
        }
    }

    if (currentIndex == (1 << n)) {
        return;
    }

    generateCombinations(n, currentIndex + 1, currentWeight + items[currentIndex].weight, 
                         currentValue + items[currentIndex].value);

    generateCombinations(n, currentIndex + 1, currentWeight, currentValue);
}

int main() {
    int n;

    printf("Enter the number of items: ");
    scanf("%d", &n);

    printf("Enter the knapsack capacity: ");
    scanf("%d", &knapsackCapacity);

    printf("Enter the weight and value of each item:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d %d", &items[i].weight, &items[i].value);
    }

    generateCombinations(n, 0, 0, 0);

    printf("Maximum total value: %d\n", maxTotalValue);
    printf("Selected items:\n");

    for (int i = 0; i < n; i++) {
        if (selectedItems[i]) {
            printf("Item %d (Weight: %d, Value: %d)\n", i + 1, items[i].weight, items[i].value);
        }
    }

    return 0;
}
