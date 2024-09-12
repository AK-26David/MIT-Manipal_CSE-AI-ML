#include<stdio.h>
int main()
{
    int a[100][100],b[100][100],i,j,m,n,p,q;
    printf("Enter the order of the matrix:");
    scanf("%d%d",&m,&n);
    printf("Enter the elements in the matrix");
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            scanf("%d",&a[i][j]);
        }
    }
     printf("Enter the order of the matrix:");
    scanf("%d%d",&p,&q);
    printf("Enter the elements in the matrix");
    for(i=0;i<p;i++)
    {
        for(j=0;j<q;j++)
        {
            scanf("%d",&a[i][j]);
        }
    }
    if(m==p&&n==q)
    {
        printf("The order of the matrix is the same.");
    }
    else
    {
        printf("The order is not the same.");
    }
    return 0;
}