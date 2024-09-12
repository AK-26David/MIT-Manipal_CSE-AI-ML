#include<stdio.h>
int printeven(int ll,int ul)
{
    if(ll>ul)
    {
        return;
    }
    if(ll%2==0)
    {
        printf("%d\t",ll);
    }
    printeven(ll+1,ul);
}
int main()
{
    int ll,ul;
    printf("Enter the lower limit and upper limit respectively:");
    scanf("%d%d",&ll,&ul);
    int even = printeven(ll,ul);
    return 0;
}