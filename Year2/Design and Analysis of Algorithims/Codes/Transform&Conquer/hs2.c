#include <stdio.h>

// Function to swap two elements in an array
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Function to perform heapify (max heap) at given index
void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    // Check if left child is larger than root
    if (left < n && arr[left] > arr[largest])
        largest = left;

    // Check if right child is larger than current largest
    if (right < n && arr[right] > arr[largest])
        largest = right;

    // If largest is not root
    if (largest != i) {
        swap(&arr[i], &arr[largest]);
        // Recursively heapify the affected subtree
        heapify(arr, n, largest);
    }
}

// Function to perform heap sort
void heapSort(int arr[], int n) {
    // Build max heap (top-down approach)
    for (int i = 1; i < n; i++) {
        // Fix max heap property upwards from current element
        int child = i;
        while (child > 0) {
            int parent = (child - 1) / 2;
            if (arr[child] > arr[parent]) {
                swap(&arr[child], &arr[parent]);
                child = parent; // Move up to the parent
            } else {
                break; // Max heap property is satisfied
            }
        }
    }

    // Extract elements from heap one by one
    for (int i = n - 1; i > 0; i--) {
        // Move current root to end
        swap(&arr[0], &arr[i]);
        // Restore max heap property on the reduced heap
        heapify(arr, i, 0);
    }
}

// Function to print an array
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

// Main function to test heap sort
int main() {
    int arr[] = {12, 11, 13, 5, 6, 7};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Array before sorting: \n");
    printArray(arr, n);

    heapSort(arr, n);

    printf("Array after sorting: \n");
    printArray(arr, n);

    return 0;
}
