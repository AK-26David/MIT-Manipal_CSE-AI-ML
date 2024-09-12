#include<stdio.h>
int main()
{
    int a[100],i,n;
    printf("Enter the dimensions of the array:");
    scanf("%d",&n);
    printf("Enter the elements in the array:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("The number is:");
    for(i=0;i<n;i++)
    {
        printf("%d",a[i]);
    }
    return 0;
}