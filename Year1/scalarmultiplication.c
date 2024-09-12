#include<stdio.h>
int main()
{
    int k,a[100][100],b[100][100],i,j,m,n;
    printf("Enter the order of the matrix:");
    scanf("%d%d",&m,&n);
    printf("Enter the elements in the matrix:");
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            scanf("%d",&a[i][j]);
        }
    }
    printf("Enter scalar value:");
    scanf("%d",&k);
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            b[i][j]=k*a[i][j];
        }
    }
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            printf("%d\t",b[i][j]);
        }
        printf("\n");
    }
    return 0;
}