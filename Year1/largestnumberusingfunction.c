#include<stdio.h>
int largest(int a[n],int n)
{
    int i,temp;
    for(i=0;i<n;i++)
    {
        if(a[i]>a[i+1])
        {
            temp=a[i];
            a[i]=a[i+1];
            a[i+1]=temp;
        }
    }
}
int main()
{
    int a[100],i,n;
    printf("Enter the dimensions of the array");
    scanf("%d",&n);
    printf("Enter the elements in the array:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    largest(a[n],n);
    printf("The sorted array is %d\t",a[i]);
    return 0;
}