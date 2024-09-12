#include<stdio.h>
int main()
{
   int a[100],i,n;
   printf("Enter the dimension of the array");
   scanf("%d",&n);
   printf("Enter the elements in the array:");
   for(i=0;i<n;i++)
   scanf("%d",&a[i]);
   for(i=0;i<n;i++)
   if(a[i]<0)
   {
    printf("%d\t",a[i]);
   }
   return 0;
}