#include<stdio.h>
int main()
{
    int a[100],i,n,count,j,freq[200];
    printf("Enter the dimensions of the matrix:");
    scanf("%d",&n);
    printf("Enter the elements of the matrix:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
        freq[i]=-1;
    }
    for(i=0;i<n;i++)
    {
        count=1;
        for(j=i+1;j<n;j++)
        {
            if(a[i]==a[j])
            {
                count++;
                freq[j]=0;
            }
        }
        if(freq[i]!=0)
        {
            freq[i]=count;
        }
    }
    for(i=0;i<n;i++)
    {
        if(freq[i]==1)
        {
            printf("%d\t",a[i]);
        }
    }
    return 0;
}