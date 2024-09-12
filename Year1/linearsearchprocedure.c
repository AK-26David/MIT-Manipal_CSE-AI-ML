#include<stdio.h>
int main()
{
    int found,a[10],i,n,key,pos;
    found=0;
    //Inputting an Array//
    printf("Enter the dimension of the array");
    scanf("%d",&n);
    printf("Enter the elements in the array:");
    for(i=0;i<n;i++)
    scanf("%d",&a[i]);
    //To search the element//
    printf("Enter the element to be searching:");
    scanf("%d",&key);
    for(i=0;i<n;i++);
    {
        //To compare//
        if(a[i]==key)
        {
            found=1;
            pos=i+1;
        }
    }
    if(found==1)
    printf("Element found in %d position",pos);
    else
    printf("Not found");
    return 0;
}