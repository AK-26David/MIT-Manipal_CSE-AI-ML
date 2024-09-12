#include<stdio.h>
void selectionSort(int a[],int n)
{
    int i,j,minIndex,temp;
    for(i =0;i<n;i++)
    {
        minIndex=i;
        for(j=i+1;j<n;j++)
        {
            if(a[j]<a[minIndex])
            {
                minIndex=j;
            }
        }
        if(minIndex!=i)
        {
            temp=a[i];
            a[i]=a[minIndex];
            a[minIndex]=temp;
        }
    }
}
int main()
{
    int a[100],i,n;
    printf("Enter the number of elements:");
    scanf("%d",&n);
    for(i=0;i<n;i++)
    {
        printf("Enter element:");
        scanf("%d",&a[i]);
    }
    selectionSort(a,n);
    for(i=0;i<n;i++)
    {
        printf("%d",a[i]);
    }
    return 0;
}