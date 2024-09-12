#include<stdio.h>
int main()
{
    int a[100][100],i,j,m,n,sum1,sum2;
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
        for(i=0;i<m;i++)
        {
            for(j=0;j<n;j++)
            {
                if(i==j)
                {
                    sum1+=a[i][j];
                }
            }
        }
        printf("The sum of the primary diagonal is %d",sum1);
        for(i=0;i<m;i++)
        {
            for(j=0;j<n;j++)
            {
                if(i+j==((n+1)-2))
                {
                    sum2+=a[i][j];
                }
            }
        }
        printf("\nThe sum of the secondary diagonal is %d",sum2);
        return 0;
}