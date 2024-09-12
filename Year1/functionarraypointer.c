#include<stdio.h>
void Compute(int *a,int size,float *sum,float *avg);
int main()
{
    int a[100],i,size;
    printf("Enter the dimensions of the array:");
    scanf("%d",&size);
    printf("Enter the elements in the array:");
    for(i=0;i<size;i++)
    {
        scanf("%d",&a[i]);
    }
    float sum,average;
    compute(size,a,&sum,&average);
    printf("%f",sum);
    printf("%f",average);
}
void Compute(int *a,int size,float *sum,float *avg)
{
    int i;
    *sum=0;
    for(i=0;i<size;i++)
    {
        *sum+=a[i];
    }
    *avg=*sum/size;
}