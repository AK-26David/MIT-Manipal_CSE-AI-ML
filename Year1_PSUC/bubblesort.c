#include<stdio.h>
int main()
{
    int a[100],i,j,n,temp;
    printf("Enter the dimension");
    scanf("%d",&n);
    printf("Enter the elements of the array");
    for(i=0;i<n;i++)
    scanf("%d",&a[i]);
    for(i=0;i<n-1;i++)
    {
        for(j=0;j=j-i-1;j++)
        {
            if(a[j]=a[j+1])
            temp=a[j];
            a[j]=a[j+1];
            a[j+1]=temp;
        }
    }
    printf("The new array is:");
    for(j=0;j=j-i-1;j++)
    printf("%d\t",a[j]);
    return 0;
}