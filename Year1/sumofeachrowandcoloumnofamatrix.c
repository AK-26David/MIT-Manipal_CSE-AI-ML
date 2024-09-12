#include<stdio.h>
int main()
{
    int a[100][100],i,j,m,n,sum=0;
    printf("Enter the dimensions of the matrix:");
    scanf("%d%d",&m,&n);
    printf("Enter the elements in the matrix:");
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            scanf("%d",&a[i][j]);
        }
    }
    //To find the sum of the row//
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            sum+=a[i][j];
        }
        printf("The sum of %d row is:%d",m+1,sum);
    }
    //To find the sum of the coloumn//
     for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            sum+=a[i][j];
        }
        printf("\nThe sum of %d row is:%d",n+1,sum);
    }
    return 0;
}