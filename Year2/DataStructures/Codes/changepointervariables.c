#include<stdio.h>
int main()
{
    int a,b,c;
    int *p,*q,*r;
    printf("Enter the 2 numbers a,b:");
    scanf("%d%d",&a,&b);
    printf("Value of a is %d, b is %d.\n",a,b);
    p=&b;
    q=p;
    r=&c;
    *q=8;
    *r=*p;
    *r=a+*q+*&c;
    printf("%d %d %d\n",a,b,c);
    printf("%d %d %d",*p,*q,*r);
    return 0;
}