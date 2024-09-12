#include<stdio.h>
int sum(int ll, int ul)
{
    if(ll==ul)
    {
        return ll;
    }
    else
    {
        return ll+sum(ll+1,ul);
    }
}
int main()
{
    int ll=1,ul;
    printf("Enter the limit of calculation:");
    scanf("%d",&ul);
    int add= sum(ll,ul);
    printf("The sum is: %d",add);
    return 0;
}