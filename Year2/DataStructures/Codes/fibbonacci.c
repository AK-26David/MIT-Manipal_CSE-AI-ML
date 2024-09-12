#include<stdio.h>
int fibo(int n)
{
    if(n==1)
    {
        return 0;
    }
    if(n==2)
    {
        return 1;
    }
    else 
    {
        return fibo(n-1)+fibo(n-2);
    }
}
int main()
{
    int n;
    printf("Enter the nth term to find its fibbonacci number:");
    scanf("%d",&n);
    int fibbonacci = fibo(n);
    printf("The %d fibbonacci term is %d",n,fibbonacci);
    return 0;
}