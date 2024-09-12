#include<stdio.h>
#include<stdlib.h>
int main()
{
    int m,n;
    int i,j;
    printf("Enter the dimensions of the first array:");
    scanf("%d",&m);
    int *a=(int *)malloc(sizeof(int));
    if(a==NULL)
    {
        return 1;
    }
    printf("Enter the dimensions of the second array:");
    scanf("%d",&n);
    int *b=(int *)malloc(sizeof(int));
    if(b==NULL)
    {
        return 1;
    }
    if(m!=n)
    {
        printf("Mathematical Operation is not possible.");
    }
    printf("Enter the elements in the first array:");
    for(i=0;i<m;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("Enter the elements in the second array:");
    for(i=0;i<n;i++)
    {
        scanf("%d",&b[i]);
    }
    int *result= (int *)malloc(sizeof(int));
     if(result==NULL)
    {
        return 1;
    }
    addArrays(a,b,result,n);
    printf("The addition of the 2 arrays is\n:");
    for(i=0;i<m;i++)
    {
        printf("%d\t",result[i]);
    }
    free(a);
    free(b);
    free(result);
    return 0;
}
void addArrays(int *a, int *b, int *result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = *(a + i) + *(b + i); 
    }
}