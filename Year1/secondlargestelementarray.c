#include<stdio.h>
int main()
{
    int a[100],i,n,secondmax,firstmax;
    printf("Enter the number of elements");
    scanf("%d",&n);
    printf("Enter the elements in the array:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    firstmax=a[0];
    for(i=1;i<n;i++)
    {
        if(a[i]>firstmax);
        secondmax=firstmax;
        firstmax=a[i];
    }
    if(a[i]>secondmax && a[i]<firstmax)
    {
    secondmax=a[i];
    }
    printf("Second largest element:%d",secondmax);
    return 0;
}