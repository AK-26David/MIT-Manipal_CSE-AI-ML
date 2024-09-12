#include <stdio.h>

struct Complex_Numbers
{
    int real;
    int imaginary;
};

void add(struct Complex_Numbers s1, struct Complex_Numbers s2);
void subtract(struct Complex_Numbers s1, struct Complex_Numbers s2);
void multiply(struct Complex_Numbers s1, struct Complex_Numbers s2);

int main()
{
    struct Complex_Numbers s1, s2;
    
    printf("Enter the real part of the first complex number:");
    scanf("%d", &s1.real);
    printf("Enter the imaginary part of the first complex number:");
    scanf("%d", &s1.imaginary);
    printf("Enter the real part of the second complex number:");
    scanf("%d", &s2.real);
    printf("Enter the imaginary part of the second complex number:");
    scanf("%d", &s2.imaginary);

    add(s1, s2);
    subtract(s1, s2);
    multiply(s1, s2);
    
    return 0;
}

void add(struct Complex_Numbers s1, struct Complex_Numbers s2)
{
    int temp_real = s1.real + s2.real;
    int temp_imaginary = s1.imaginary + s2.imaginary;

    printf("Sum of the complex numbers: %d + %di\n", temp_real, temp_imaginary);
}

void subtract(struct Complex_Numbers s1, struct Complex_Numbers s2)
{
    int temp_real = s1.real - s2.real;
    int temp_imaginary = s1.imaginary - s2.imaginary;

    printf("Difference of the complex numbers: %d + %di\n", temp_real, temp_imaginary);
}

void multiply(struct Complex_Numbers s1, struct Complex_Numbers s2)
{
    int temp_real = s1.real * s2.real - s1.imaginary * s2.imaginary;
    int temp_imaginary = s1.real * s2.imaginary + s1.imaginary * s2.real;

    printf("Product of the complex numbers: %d + %di\n", temp_real, temp_imaginary);
}
