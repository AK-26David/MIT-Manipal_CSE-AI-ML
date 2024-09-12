#include<stdio.h>
int printNaturalNumbers(int ll,int ul)
{
    if(ll>ul)
    {
        return;
    }
    else
    {
        printf("%d\t",ll);
        printNaturalNumbers(ll+1,ul);
    }
}
int oddnumbers(int ll,int ul)
{
     if(ll>ul)
    {
        return;
    }
    if(ll%2==0)
    {
        return ll;
    }
    else
    {
        printf("%d\t",ll);
        oddnumbers(ll+2,ul);
    }
}
int oddsum(int ll,int ul)
{
    if(ll==ul)
    {
        return ll;
    }
    if(ll%2==0)
    {
        return ll;
    }
    else
    {
        return ll+oddsum(ll+2,ul);
    }
}
int main()
{
    int ll,ul;
    printf("Enter the range for calculation:");
    scanf("%d%d",&ll,&ul);
    int nn=printNaturalNumbers(ll,ul);
    printf("\n");
    int odd=oddnumbers(ll,ul);
    printf("\n");
    int sum=oddsum(ll,ul);
    printf("The sum of the odd numbers is %d",sum);
    return 0;
}