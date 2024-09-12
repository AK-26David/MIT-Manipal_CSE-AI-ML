#include<stdio.h>
int main()
{
    int a,b;
    int *p,*q;
    printf("Enter the value of a and b respectively:");
    scanf("%d%d",&a,&b);
    p=&a;
    q=&b;
    int sum=*p+*q;
    printf("The sum is %d",sum);
    return 0;
}