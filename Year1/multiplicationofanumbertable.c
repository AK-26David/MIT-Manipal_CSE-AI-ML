#include<stdio.h>
int main()
{
    int i,j,m,n,prod;
    printf("Enter the limits:");
    scanf("%d%d",&m,&n);
    for(i=1;i<m;i++)
    {
        for(j=1;j<n;j++)
        {
            prod=i*j;
            printf("%d*%d=%d\n",i,j,prod);
        }
    }
    printf("\t");
    return 0;
}