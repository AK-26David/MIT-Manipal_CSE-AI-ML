#include<stdio.h>
int divide(int a,int b)
{
    if(a==0)
    {
        return 0;
    }
    else if(b==0)
    {
        printf("Zero division error!!!");
    }
    else if(a-b==0)
    {
        return 1;
    }
    else
    {
        return (1+divide(a-b,b));
    }
}
int main()
{
    int num1,num2;
    printf("Enter number 1 and number 2 respectively:");
    scanf("%d%d",&num1,&num2);
    int quotient=divide(num1,num2);
    printf("%d",quotient);
    return 0;
}