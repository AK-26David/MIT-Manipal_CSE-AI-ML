#include<stdio.h>
int main()
{
    int limit,i;
    printf("Enter the limit to print the natural numbers:");
    scanf("%d",&limit);
    for(i=0;i<=limit;i++)
    {
        printf("%d\n",i);
    }
    return 0;
}