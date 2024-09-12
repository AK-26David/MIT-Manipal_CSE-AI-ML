#include<stdio.h>
int main()
{
    int a[100][100],m,n,i,j,temp;
    printf("Enter the number of rows and coloumns in the 1st array:");
    scanf("%d%d",&m,&n);
    printf("Enter the elements:");
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
            printf("%d\t",a[i][j]);
        }
        printf("\n");
    }
    printf("\n\n\n");
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            if(i==j)
            {
                temp=a[i][j];
                a[i][j]=a[n-1-i][j];
                a[n-1-i][j]=temp;
            }
        }
    }
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            printf("%d\t",a[i][j]);
        }
        printf("\n");
    }
    return 0;
}