#include<stdio.h>
int max(int a,int b)
{
    int max;
    return(a>b)?a:b;
}
int main()
{
    int a,b,maximum;
    printf("Enter the 2 numbers a and b");
    scanf("%d%d",&a,&b);
    maximum=max(a,b);
    printf("%d",maximum);
    return 0;
}