#include<stdio.h>
int main()
{
    int num1,num2,sum,diff,prod,quot,result;
    char op;
    printf("Enter the 2 numbers a and b");
    scanf("%d%d",&num1,&num2);
    printf("Choose the mode of operation:");
    scanf("%c",&op);
    switch(op)
    {
        case '+': 
            result = num1 + num2;
            break;

        case '-': 
            result = num1 - num2;
            break;

        case '*': 
            result = num1 * num2;
            break;

        case '/': 
            result = num1 / num2;
            break;

        default: 
            printf("Invalid operator");
    }

    /* Prints the result */
    printf("%.2f %c %.2f = %.2f", num1, op, num2, result);

    return 0;
}