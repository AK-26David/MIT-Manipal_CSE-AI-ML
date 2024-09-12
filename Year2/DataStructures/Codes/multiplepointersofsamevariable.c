#include<stdio.h>
int main()
{
    int a;
    int *p=&a;
    int *q=&a;
     int *r=&a;
     printf("Enter a number :");
     scanf("%d",&a);
     printf("%d\n%d\n%d\n",*p,*q,*r);
     return 0;
}