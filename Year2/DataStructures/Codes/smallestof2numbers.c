#include <stdio.h>

int* findSmaller(int* num1, int* num2) {
    if (*num1 < *num2) {
        return num1;
    } else {
        return num2;
    }
}

int main() {
    int num1, num2;

    printf("Enter the first number: ");
    scanf("%d", &num1);

    printf("Enter the second number: ");
    scanf("%d", &num2);

    int* smallerAddress = findSmaller(&num1, &num2);

    printf("The smaller number is: %d\n", *smallerAddress);

    return 0;
}
