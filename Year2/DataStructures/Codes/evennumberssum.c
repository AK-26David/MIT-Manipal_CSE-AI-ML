#include<stdio.h>
int evensum(int ll,int ul)
{
    if(ll%2!=0 && ul%2!=0)
    {
        return 0;
    }
    if(ll==ul)
    {
        return ll;
    }
    else
    {
        return ll+evensum(ll+2,ul);
    }
}
int main()
{
    int ll,ul;
    printf("Enter the lower limit and upper limit respectively in even terms:");
    scanf("%d%d",&ll,&ul);
    int sum=evensum(ll,ul);
    printf("The sum is %d",sum);
    return 0;
}