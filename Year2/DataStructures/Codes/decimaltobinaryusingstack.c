#include<stdio.h>

void DecimaltoBinary(int num);

int main() {
    int num;
    printf("Enter a decimal number to be converted to a binary number: ");
    scanf("%d", &num);
    if (num < 0) {
        printf("Please enter a positive number!!!!");
    } else {
        DecimaltoBinary(num);
    }
    return 0;
}

void DecimaltoBinary(int num) {
    int size = 32;
    int Stack[size];
    int top = -1;
    int binarydigit;
    
    if (num == 0) {
        printf("The binary representation is 0!!!");
    }
    
    // Compute binary representation and push it onto the stack
    while (num > 0) {
        binarydigit = num % 2;
        Stack[++top] = binarydigit;
        num = num / 2;
    }
    
    // Display the binary representation
    printf("The binary representation is: ");
    while (top >= 0) {
        printf("%d", Stack[top--]);
    }
}
