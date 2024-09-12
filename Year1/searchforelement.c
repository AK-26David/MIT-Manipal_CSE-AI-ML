#include<stdio.h>
int main()
{
    int a[100],i,n,key,pos,found;
    found=0;
    printf("Enter the dimension of the array");
    scanf("%d",&n);
    printf("Enter the elements in the array");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("Enter the search element");
    scanf("%d",&key);
    for(i=0;i<n;i++)
    {
         if(a[i]=key)
        {
            found=1;
            pos=i+1;
            break;
        }
    }
    if(found==1)
    {
        printf("Element found in %d position",pos);
    }
    else
        printf("Not found");
        return 0;
}