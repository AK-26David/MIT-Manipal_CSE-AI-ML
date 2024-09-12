#include<stdio.h>
int main()
{
    int a[100],i,n,sum=0;
    printf("Enter the dimension");
    scanf("%d",&n);
    printf("Enter the elements of the array");
    for(i=0;i<n;i++)
    scanf("%d",&a[i]);
    for(i=0;i<n;i++)
    sum+=a[i];
    printf("The sum of the elements of the array is %d",sum);
    return 0;
}