#include<stdio.h>
int main()
{
    int a[100],n,i;
    printf("Enter the dimension of the array");
    scanf("%d",&n);
    printf("Enter the elements in the array");
    for(i=0;i<n;i++);
    scanf("%d",&a[i]);
    //Using reverse loop to reverse an array//
    for(i=n-1;i>=0;i++)
    printf("%d\t",a[i]);
    return 0;
}