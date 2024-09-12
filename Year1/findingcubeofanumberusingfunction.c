#include<stdio.h>
int cube(int num)
{
    int c;
    c=num*num*num;
    printf("%d",c);
    return c;
}
int main()
{
    int num,c;
    printf("Enter the number:");
    scanf("%d",&num);
    cube(num);
    return 0;
}