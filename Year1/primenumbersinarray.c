#include<stdio.h>
int main()
{
    int a[100],i,n,j,flag=0;
    printf("Enter the dimensions of the array:");
    scanf("%d",&n);
    printf("Enter elements in the array:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    for(i=0;i<n;i++)
    {
        for(j=2;j<a[i];j++)
        {
            if(a[i]%j==0)
            {
                flag=1;
                break;
            }
        }
        if(flag==0)
        {
            printf("%d",a[i]);
        }
    }
    return 0;
}