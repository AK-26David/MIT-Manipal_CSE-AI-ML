#include <stdio.h>
#include <stdlib.h>

void rev(int a[], int n) {
    int i;
    int *p = a;
    int *q = a + n - 1; // Initialize q to the last element
    for (i = 0; i < n / 2; i++) {
        int temp = *p;
        *p = *q;
        *q = temp;
        p++;
        q--;
    }
}

int main() {
    int n, i;
    printf("Enter the dimensions of the array: ");
    scanf("%d", &n);

    int *a = (int *)calloc(n, sizeof(int)); // Allocate memory for 'a'

    if (a == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    printf("Enter the elements in the array: ");
    for (i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }

    printf("The reverse array is: ");
    rev(a, n);

    for (i = 0; i < n; i++) {
        printf("%d\t", a[i]);
    }

    // Free the allocated memory
    free(a);

    return 0;
}
