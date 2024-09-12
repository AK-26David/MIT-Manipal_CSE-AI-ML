#include<stdio.h>
int main()
{
    int a[100][100],b[100][100],transpose,i,j,m,n;
    printf("Enter the dimensions of the matrix:");
    scanf("%d%d",&m,&n);
    printf("Enter the elements of the matrix:");
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
            b[j][i]=a[i][j];
        }
    }
    int flag = 0;
    for(i=0;i<m;i++)
    {
        for(j=0;j<m;j++)
        {
            if(b[i][j]!=a[i][j])
            {
                printf("Not symmetric");
                flag = 1;
                break;
            }
        }
    }
    if(flag==0)
    {
        printf("Is symmetric");
    }
    return 0;
}