#include<stdio.h>
int main()
{
    int a[100],i,n,odd=0,even=0;
    printf("Enter the dimensions of the array");
    scanf("%d",&n);
    printf("Enter the elements in the array");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    for(i=0;i<n;i++)
    {
        if(a[i]%2==0)
        {
            even++;
        }
        printf("The even matrix:%d",a[i]);
        if(a[i]%2!=0)
        {
            odd++;
        }
        printf("\nThe odd matrix:%d",a[i]);
    }
    printf("\nThe total number of even elements in the array:%d",even);
    printf("\nThe total number of odd elements in the array:%d",odd);
    return 0;
}