#include<stdio.h>
void replace(int a[],int large)
{
    int i,n,max=a[0];
    for(i=1;i<n;i++)
    {
        if(a[i]>max)
        {
            max=a[i];
        }
    }
    for(i=0;i<n;i++)
    {
        if(a[i]%2!=0)
        {
            a[i]=max;
        }
    }
}
int main()
{
    int n, a[20], x, y,i;
    printf("Enter the limit \n");
    scanf("%d",&n);
    printf("Enter the values: \n");
    for (i=0; i<n; i++)
    {
        scanf("%d",&a[i]);
    }
    replace(a, n);
    printf("The array after replacement is\n");
    for (i=0; i<n; i++)
    printf("%d \n",a[i]);
    return 0;
}