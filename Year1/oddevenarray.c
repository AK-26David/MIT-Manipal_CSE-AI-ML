#include<stdio.h>
int main()
{
    int a[100],i,n,odd=0,even=0;
    printf("Enter the number of elements in the array");
    scanf("%d",&n);
    printf("Enter the elements in the array");
    for(i=0;i<n;i++)
    scanf("%d",&a[i]);
    printf("Even numbers in the array are: ");
    for(int i=0;i<n;i++)
    {
        if(a[i]%2==0)
        printf("%d ",a[i]);
    }
        printf("\nOdd numbers in the array are: ");
    for(int i=0;i<n;i++)
    {
        if(a[i]%2==1)
        printf("%d ",a[i]);
    }
    return 0;
}