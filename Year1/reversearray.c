#include<stdio.h>
int main()
{
    int a[20],m,i,temp;
    printf("Enter the dimension of the array");
    scanf("%d",&m);
    printf("Enter the elements in the array");
    for(i=0;i<m;i++);
    scanf("%d",&a[i]);
    printf("Array is:");
    for(i=0;i<m;i++)
    printf("%d\t",a[i]);
    //To reverse an array//
    for(i=0;i<m/2;i++)
    {
        temp=a[i];
        a[i]=a[m-i-1];
        a[m-i-1]=temp;
    }
    printf("Reversed array is:");
    for(i=0;i<m;i++)
    printf("%d\t",a[i]);
    return 0;
}
    