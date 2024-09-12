#include<stdio.h>
void replace(int a[],int n);
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
    replace(a,n);
    printf("The array after replacement:\n");
    for(i=0;i<n;i++)
    {
        printf("%d\t",a[i]);
    }
    return 0;
}
void replace(int a[],int n)
{
    int i,max=a[0];
    for(i=1;i<n;i++)
    {
        if(a[i]>max)
        {
            a[i]=max;
        }
    }
    for(i=0;i<n;i++)
    {
        if(a[i]%2!=0)
        {
            a[i]=max;
        }
    }
}
