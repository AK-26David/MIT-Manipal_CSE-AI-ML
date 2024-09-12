#include<stdio.h>
void bubbleSort(int a[], int n)
{
    int i,j,temp;
    for(i=0;i<n;i++)
    {
        for(j=i+1;j<n;j++)
        {
            if(a[i]>a[j])
            temp=a[i];
            a[i]=a[j];
            a[j]=temp;
        }
    }
}
int main()
{
    int a[100],i,n;
    printf("Enter the number of elements in the array:");
    scanf("%d",&n);
    for(i=0;i<n;i++)
    {
        printf("Enter the element:");
        scanf("%d",&a[i]);
    }
    bubbleSort(a,n);
    for(i=0;i<n;i++)
    {
        printf("%d",a[i]);
    }
    return 0;
}