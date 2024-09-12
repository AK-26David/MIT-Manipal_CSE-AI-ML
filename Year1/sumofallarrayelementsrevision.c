#include<stdio.h>
int main()
{
    int a[100],sum=0,i,n;
    printf("Enter the dimension of the array:");
    scanf("%d",&n);
    printf("Enter the elements in the array");
    for(i=0;i<n;i++)
    scanf("%d",&a[i]);
    for(i=0;i<n;i++)
    sum+=a[i];
    printf("%d",sum);
    return 0;
}