#include<stdio.h>
int multiply(int a, int b)
{
    if(a==0 || b==0)
    {
        return 0;
    }
    if(a==1)
    {
        return b;
    }
    if(b==1)
    {
        return a;
    }
    else
    {
        int y= multiply(a,b-1);
        return y+a;
    }
}
int main()
{
    int a,b;
    printf("Enter two numbers a and b for calculation respectively:");
    scanf("%d%d",&a,&b);
    int prod=multiply(a,b);
    printf("The product is:%d",prod);
    return 0;
}