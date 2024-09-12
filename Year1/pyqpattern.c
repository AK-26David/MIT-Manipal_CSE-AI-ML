#include<stdio.h>
int main()
{
    int i,j,n,m,k;
    printf("Enter the number of rows:");
    scanf("%d",&n);
    for(i=0;i<n;i++)
    {
        for(j=0;j<i;j++)
        {
            printf(" ");
        }
        for(k=0;k<m;k++)
        {
            printf("*");
        }
        m--;
        printf("\n");
    }
    return 0;
}