#include<stdio.h>
int main()
{
    int a[100],i,n;
    printf("Enter the dimensions of the 1d array:");
    scanf("%d",&n);
    printf("Enter the elements in the 1d array:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    for(i=n-1;i>=0;i--)
    {
        printf("\n\n%d\n",a[i]);
    }
    return 0;
}