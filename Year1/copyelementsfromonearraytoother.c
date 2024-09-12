#include<stdio.h>
int main()
{
    int a[100],b[100],i,m,n;
    printf("Enter the number of elements in the array");
    scanf("%d",&m);
    printf("Enter the elements in the array");
    for(i=0;i<m;i++)
    scanf("%d",&a[i]);
    printf("Enter the order of the 2nd array");
    scanf("%d",&n);
    for(i=0;i<m;i++)
    if(m==n)
    {
        b[i]=a[i];
    }
    printf("The new array is %d\t",b[i]);
    return 0;
}