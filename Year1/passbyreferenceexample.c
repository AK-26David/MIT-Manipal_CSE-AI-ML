#include<stdio.h>
int swap(int*x,int*y)
{
    int t=*x;
    *x=*y;
    *y=t;
    return t;
}
int main()
{
    int a=5,b=7;
    swap(&a,&b);
    printf("after swapping a=%d and b=%d",a,b);
    return 0;
}