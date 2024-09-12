#include<stdio.h>
int main()
{
    int a[100][100],i,j,m,n;
    printf("Enter the number of rows and coloumns in the array:");
    scanf("%d%d",&m,&n);
    printf("Enter the elements in the array:");
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        scanf("%d",&a[i][j]);
    }
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        printf("%d\t",a[i][j]);
        printf("\n");
    }
    return 0;
}