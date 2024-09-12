#include<stdio.h>
int main()
{
    int a[3][3],i,j,m,n,det;
    printf("Enter the elements of the array:");
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            scanf("%d",&a[i][j]);
        }
    }
    det=a[1][1]*((a[2][2]*a[3][3])-(a[3][2]*a[2][3]))-a[1][2]*((a[2][1]*a[3][3])-(a[3][1]*a[2][3]))+a[1][3]*((a[2][1]*a[3][2])-(a[3][1]*a[2][2]));
    printf("Determinant :%d",det);
    return 0;
}