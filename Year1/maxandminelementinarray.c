#include<stdio.h>
int main()
{
    int a[100],i,n,max,min;
    printf("Enter the order of the array:");
    scanf("%d",&n);
    printf("Enter the elements in the array");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    max=a[0];
    min=a[0];
    for(i=1;i<n;i++)
    {
        if(a[i]>max)
        {
        max=a[i];
        }
        if(a[i]<min)
        {
        min=a[i];
        }
    }
    printf("Maximum:%d",max);
    printf("Minimum:%d",min);
    return 0;
}