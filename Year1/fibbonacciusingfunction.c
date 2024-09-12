#include<stdio.h>
int fibb(int num)
{
    int i,a=0,b=1,c=0;
    printf("The fibbonacci terms are:");
    for(i=0;i<=num;i++)
    {
        printf("%d\n",c);
        a=b;
        b=c;
        c=a+b;
    }
}
int main()
{
    int num;
    printf("Enter the limit for the sequence:");
    scanf("%d",&num);
    fibb(num);
    return 0;
}