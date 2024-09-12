#include<stdio.h>
int add(int a[20],int n)
{
    int sum=0,i;
    for(i=0;i<n;i++)
    {
        if(a[i]%2==0)
        {
            sum+=a[i];
        }
    }
    return sum;
}
int main()
{
    int a[20],i,n;
    printf("Enter the dimensions of the matrix:");
    scanf("%d",&n);
    printf("Enter the elements in the matrix:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    add(a[20],n);
    printf("The sum:%d",add(a[20],n));
    return 0;
}