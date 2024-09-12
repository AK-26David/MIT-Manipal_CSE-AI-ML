#include<stdio.h>
#include<math.h>
int fact(int n)
{
    if(n==0 || n==1)
    {
        return 1;
    }
    else
    {
        return n*fact(n-1);
    }
}
int main()
{
    int n;
    printf("Enter the number to calculate the factorial of:");
    scanf("%d",&n);
    int factorial = fact(n);
    printf("The factorial of %d is %d.",n,factorial);
    return 0;
}