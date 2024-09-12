#include<stdio.h>
int main()
{
    int a[100][100],b[100][100],i,j,m,n,p,q;
    printf("Enter the order of matrix 1:");
    scanf("%d%d",&m,&n);
    printf("Enter the order of matrix 2:");
    scanf("%d%d",&p,&q);
    if(m!=p&&n!=q)
    {
        printf("Matrices cannot be compared");
    }
    else
    {
        printf("Enter the elements in matrix 1:");
        for(i=0;i<m;i++)
        {
            for(j=0;j<n;j++)
            {
                scanf("%d",&a[i][j]);
            }
        }
        printf("Enter the elements in matrix 2:");
        for(i=0;i<m;i++)
        {
            for(j=0;j<n;j++)
            {
                scanf("%d",&b[i][j]);
            }
        }
        for(i=0;i<m;i++)
        {
            for(j=0;j<n;j++)
            {
                if(a[i][j]==b[i][j])
                {
                printf("All elements of the matrix are equal.");
                }
                else
                {
                    printf("All elements of the matrix are not equal.");
                }
            }
        }
    }
    return 0;
}