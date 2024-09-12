#include<stdio.h>
int reverse(int num)
{
    int rev=0;
    int digit;
    while(num!=0)
    {
        digit=num%10;
        rev=(10*rev)+digit;
        num=num/10;
    }
    return (rev);
}
int main()
{
    int n,r;
    printf("Enter a number n:");
    scanf("%d",&n);
    r=reverse(n);
    printf("The reversed number is %d",r);
    return 0;
}