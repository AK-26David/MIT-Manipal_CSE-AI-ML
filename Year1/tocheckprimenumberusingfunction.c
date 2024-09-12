#include<stdio.h>
int isPrime(int a)
{
    int num;
    for(num=2;num<=a/2;a++)
    {
        if (num%a==0)
        {
            return 0;
        }
    }
    return 1;
}
void main()
{
    int a;
    printf("Enter the number to check:");
    scanf("%d",&a);
    if(isPrime(a))
    {
        printf("Number is composite");
    }
    else
    {
        printf("Number is prime");
    }
}
