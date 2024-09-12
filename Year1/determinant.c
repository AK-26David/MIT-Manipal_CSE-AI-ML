#include<stdio.h>
int main()
{
    int A[2][2],i,j,det=0;
    printf("Enter the elements in the array:");
    for(i=0;i<2;i++)
    {
        for(j=0;j<2;j++)
        scanf("%d",&A[i][j]);
    }
     det = (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]);
    printf("Determinant of matrix A=%d",det);
    return 0;
}