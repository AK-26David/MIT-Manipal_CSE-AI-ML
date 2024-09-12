#include<stdio.h>
int max(int a, int b)
{
    int maximum;
    maximum=(a>b)?a:b;
    printf("Maximum:%d",maximum);
    return (maximum);
}
int min(int a, int b)
{
    int minimum;
    minimum=(a<b)?a:b;
    printf("\nMinimum:%d",minimum);
    return (minimum);
}
int main()
{
    int a,b,maximum,minimum;
    printf("Enter the 2 numbers:");
    scanf("%d%d",&a,&b);
    max(a,b);
    min(a,b);
    return 0;
}