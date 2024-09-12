#include<stdio.h>
int checkprime(int n);
int main()
{
    int m,n,i,flag;
    printf("Enter the limits:");
    scanf("%d%d",&m,&n);
    if(m>n)
    {
        m=m+n;
        n=m-n;
        m=m-n;
    }
    printf("Prime numbers between %d and %d are:",m,n);
    for(i=m+1;i<n;++i)
    {
        flag=checkprime(i);
        if(flag==1)
        printf("%d\n",i);
    }
    return 0;
}
int checkprime(int n)
{
int j,flag=1;
for(j=2;j<=n/2;j++)
{
if(n%j==0)
{
    flag=0;
    break;
}
}
return flag;
}
