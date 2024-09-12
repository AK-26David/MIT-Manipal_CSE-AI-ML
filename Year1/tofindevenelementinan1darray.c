#include<stdio.h>
int main()
{
    int a[100],i,n,even;
    printf("Enter the dimensions in the array:");
    scanf("%d",&n);
    printf("Enter the elements in the array:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("The even elements in the array are:\n\n");
    for(i=0;i<n;i++)
    {
        if(a[i]%2==0)
        {
            printf("%d\t",a[i]);
        }
    }
    return 0;
}