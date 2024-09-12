#include<stdio.h>
int main()
{
    int a[10],i;
    int *ptr = a;
    printf("Enter 10 elements in the array");
    for(i=0;i<10;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("The value of the 1st index element is %d\n",a[1]);
    printf("The address of the first index element is %p\n",&a[1]);
    //Using Array Pointers to find address//
    printf("Using pointer the address of the first index element is %p\n",a+1);
    printf("Using derefrencing operator the address of the first index element is %p",*(&a+1));
     printf("Using derefrencing operator the address of the first index element is %p",*(a+2));// adding 1 byte//
    return 0;
}