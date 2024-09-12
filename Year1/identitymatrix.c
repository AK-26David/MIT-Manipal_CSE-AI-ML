#include<stdio.h>
int main()
{
    int a[100][100],i,j,m,n,flag;
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
    //To check identity matrix//
    flag=1;
    //To check diagonal elements//
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            if(i==j&&a[i][j]!=1)
            {
            flag=0;
            }
            //To check other matrix elements//
            if(i!=j&&a[i][j]!=0)
            {
                flag=0;
            }
        }
    }
    if(flag==1)
    {
    printf("The matrix is an identity matrix.");
    }
    else
    {
        printf("The matrix is not an identity matrix.");
    }
    return 0;
}