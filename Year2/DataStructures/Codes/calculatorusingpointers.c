#include <stdio.h>

int *add(int *a, int *b)
{
    int sum = *a + *b;
    return sum;
}

int *sub(int *a, int *b)
{
    int diff = *a - *b;
    return diff;
}

int *multiply(int *a, int *b)
{
    int prod = (*a) * (*b);
    return prod;
}

int *divide(int *a, int *b)
{
    int quo = (*a) / (*b);
    return quo;
}

int main()
{
    int a, b;
    int *ptr1 = &a;
    int *ptr2 = &b;
    printf("Enter two numbers for calculation:");
    scanf("%d%d", &a, &b);
    int sum = add(ptr1, ptr2);
    int diff = sub(ptr1, ptr2);
    int mul = multiply(ptr1, ptr2);
    int div = divide(ptr1, ptr2);
    printf("Sum: %d\n", sum);
    printf("Difference: %d\n", diff);
    printf("Product: %d\n", mul);
    printf("Quotient: %d\n", div);
    return 0;
}
