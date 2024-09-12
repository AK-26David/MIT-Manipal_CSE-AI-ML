#include <stdio.h>

int main() {
    int a[100][100], b[100][100], c[100][100], i, j, k, m, n, p, q,count=0;

    printf("Enter the dimensions of matrix A:");
    scanf("%d%d", &m, &n);

    printf("Enter the dimensions of matrix B:");
    scanf("%d%d", &p, &q);

    if (n != p) {
        printf("Matrix Multiplication not possible. Process Terminated!!!");
        return 1;  // Return an error code
    }

    // Input matrix A
    printf("Enter the elements of matrix A:\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            scanf("%d", &a[i][j]);
        }
    }

    // Input matrix B
    printf("Enter the elements of matrix B:\n");
    for (i = 0; i < p; i++) {
        for (j = 0; j < q; j++) {
            scanf("%d", &b[i][j]);
        }
    }

    // Initialize result matrix C to zeros
    for (i = 0; i < m; i++) {
        for (j = 0; j < q; j++) {
            c[i][j] = 0;
        }
    }

    // Matrix multiplication
    for (i = 0; i < m; i++) {
        for (j = 0; j < q; j++) {
            for (k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
                count++;
            }
        }
    }

    // Print the result matrix C
    printf("Result of matrix multiplication (C = A * B):\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < q; j++) {
            printf("%d \t ", c[i][j]);
        }
        printf("\n");
    }
    printf("Opcount:",count);
    return 0;
}
