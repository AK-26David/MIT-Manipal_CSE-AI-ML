#include<stdio.h>
int main()
{
    int a[100],i,n,even,odd;
    printf("Enter the dimensions of the 1d array:");
    scanf("%d",&n);
    printf("Enter the elements in the array:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    //Sorting of even elements//
    for(i=0;i<n;i++)
    {
        if(a[i]%2==0)
        {
        printf("%d\t",a[i]);
        }
    }
    printf("\n\n");
    //Sorting of odd elements//
    for(i=0;i<n;i++)
    {
        if(a[i]%2!=0)
        {
            printf("%d\t",a[i]);
        }
    }
    return 0;
}