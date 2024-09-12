#include<stdio.h>
int product(int a,int b)
{
    int c;
    c=a*b;
    return c;
}
int main()
{
    int a,b,c;
    printf("Enter the 2 numbers a and b");
    scanf("%d%d",&a,&b);
    c=product(a,b);
    printf("%d",c);
    return 0;
}