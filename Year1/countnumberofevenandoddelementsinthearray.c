#include<stdio.h>
int main()
{
    int a[100],i,n,even=0,odd=0;
   printf("Enter the dimension of the array");
   scanf("%d",&n);
   printf("Enter the elements in the array:");
   for(i=0;i<n;i++)
   scanf("%d",&a[i]);
   for(i=0;i<n;i++)
   if(a[i]%2==0)
   {
    even++;
   }
   else
   {
    odd++;
   }
   printf("Even=%d\n",even);
   printf("Odd=%d",odd);
   return 0;
}