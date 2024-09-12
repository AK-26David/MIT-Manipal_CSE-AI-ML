#include <stdio.h>
#include <stdlib.h>

int smallest(int *a, int n); // Declare the function prototype

int main() {
    int n;
    printf("Enter the dimensions in the array:");
    scanf("%d", &n);

    // Check if memory allocation was successful
    int *a = (int *)malloc(n * sizeof(int));
    if (a == NULL) {
        return 1;
    }

    printf("Enter the elements in the array:");
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }

    int Smallest = smallest(a, n);
    printf("The smallest element is %d\n", Smallest);

    // Don't forget to free the allocated memory
    free(a);

    return 0;
}

int smallest(int *a, int n) {
    int smallest = a[0]; // Initialize smallest with the first element
    for (int i = 1; i < n; i++) {
        if (*(a + i) < smallest) { // Use pointer arithmetic
            smallest = *(a + i);
        }
    }
    return smallest;
}
