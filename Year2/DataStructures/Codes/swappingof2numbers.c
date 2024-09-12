#include<stdio.h>
void swap(int *p, int *q)
{
    int temp;
    temp=*p;
    *p=*q;
    *q=temp;
}
int main()
{
    int m,n;
    printf("Enter two numbers:");
    scanf("%d%d",&m,&n);
    printf("Before swapping m= %d and n= %d.",m,n);
    swap(&m,&n);
     printf("Before swapping m= %d and n= %d.",m,n);
    return 0;
}