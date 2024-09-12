#include<stdio.h>
int main()
{
    int a[100],i,n,negative=0;
    printf("Enter the dimensions of the array:");
    scanf("%d",&n);
    printf("Enter the elements of the array:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    for(i=0;i<n;i++)
    {
        if(a[i]<0)
        {
            negative++;
        }
    }
    printf("The number of negative elements in the array are:%d",negative);
    return 0;
}