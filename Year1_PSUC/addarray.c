#include<stdio.h>
int main()
{
    int a[100],b[100],c[100],i,m,n;
    //Array A//
    printf("Enter the number of elements in the array");
    scanf("%d",&m);
    printf("Enter the elements in the array");
    for(i=0;i<m;i++)
    scanf("%d",&a[i]);
    //Array B//
    printf("Enter the number of elements in array");
    scanf("%d",&n);
    printf("Enter the elements in the array");
    for(i=0;i<n;i++)
    scanf("%d",&b[i]);
    if(m==n)
    {
        for(i=0;i<n;i++)
        c[i]=a[i]+b[i];
        printf("The sum of the array:");
        for(i=0;i<n;i++)
        printf("%d\t",c[i]); 
    }
    else
    {
        printf("Invalid Input");
    }
}