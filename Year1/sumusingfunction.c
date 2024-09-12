#include<stdio.h>
int main()
{
    int a,b;
    printf("Enter the 2 numbers a and b:");
    scanf("%d%d",&a,&b);
    sum(a,b);  
}
int sum(int a,int b)
{
    int c;
    c=a+b;
    printf("%d",c);
    return 0;
}