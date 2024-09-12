#include<stdio.h>
int quotient(int a, int b)
{
    int q;
    q=a/b;
    printf("%d",q);
    return q;
}
int main()
{
    int a,b,q;
    printf("Enter the 2 numbers a and b");
    scanf("%d%d",&a,&b);
    quotient(a,b);
    return 0;
}