#include<stdio.h>
int main()
{
    int a[100][100],b[100][100],c[100][100],m,n,p,q,i,j;
    printf("Enter the number of rows and coloumns in the 1st array:");
    scanf("%d%d",&m,&n);
    printf("Enter the elements in the array:");
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        scanf("%d",&a[i][j]);
    }
    printf("Enter the number of rows and coloumns in the 2nd array:");
    scanf("%d%d",&p,&q);
    printf("Enter the elements in the array:");
    for(i=0;i<p;i++)
    {
        for(j=0;j<q;j++)
        scanf("%d",&b[i][j]);
    }
    if(m==p&&n==q)
    {
        for(i=0;i<m;i++)
        {
            for(j=0;j<n;j++)
            c[i][j]=a[i][j]+b[i][j];
        }
          for(i=0;i<m;i++)
          {
            for(j=0;j<n;j++)
            printf("%d\t",c[i][j]);
            printf("\n");
          }
    }else{
        printf("Impossible to add\n");
        exit(0);
    }
    return 0;
}