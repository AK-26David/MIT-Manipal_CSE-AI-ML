#include<stdio.h>
int main()
{
    int a;
    int *ptr;
    printf("Enter the value of a:");
    scanf("%d",&a);
    printf("The value of a is %d\n. The address of a is %p.\n",a,&a);
    ptr=&a;
    printf("Pointer value %p containing the address of a is %d.",ptr,*ptr);
    return 0;
}