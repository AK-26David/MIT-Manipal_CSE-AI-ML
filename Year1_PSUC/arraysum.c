#include<stdio.h>
int arraysum(int a[],int n);
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
    printf("%d",arraysum(a,n));
    return 0;
}
int arraysum(int a[],int n)
{
    int sum=0,i;
    for(i=0;i<n;i++)
    {
        sum+=a[i];
    }
    return sum;
}