#include<stdio.h>
int isEven(int num)
{
    int oddeven;
    if(num%2==0)
    {
        printf("%d is even",num);
    }
    else
    {
        printf("%d is odd",num);
    }
    return num;
}
int main()
{
    int num;
    printf("Enter the number:");
    scanf("%d",&num);
    isEven(num);
    return 0;
}